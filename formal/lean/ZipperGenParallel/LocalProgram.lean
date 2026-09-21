import ZipperGenParallel.Parloc

namespace ZipperGenParallel

universe u v

/-- Local programs over the current abstract local alphabet. -/
inductive LocProg (Lifeline : Type u) (Channel : Type v) :
    Type (max u v) where
  | eps : LocProg Lifeline Channel
  | letter (letter : LocalLetter Lifeline Channel) : LocProg Lifeline Channel
  | seq (first second : LocProg Lifeline Channel) : LocProg Lifeline Channel
  | branch
      (trueLetter falseLetter : LocalLetter Lifeline Channel)
      (thenBranch elseBranch : LocProg Lifeline Channel) :
      LocProg Lifeline Channel
  | loop
      (trueLetter falseLetter : LocalLetter Lifeline Channel)
      (body exit : LocProg Lifeline Channel) :
      LocProg Lifeline Channel
  | par {n : Nat}
      (branches : Fin n → LocProg Lifeline Channel) :
      LocProg Lifeline Channel

namespace LocProg

infixl:50 " ;;ₗ " => LocProg.seq

/-- Concatenate a finite list of local words. -/
def concatWords {Lifeline : Type u} {Channel : Type v}
    (words : List (Word (LocalLetter Lifeline Channel))) :
    Word (LocalLetter Lifeline Channel) :=
  words.foldl (· ++ ·) []

private theorem concatWords_foldl {Lifeline : Type u} {Channel : Type v}
    (acc : Word (LocalLetter Lifeline Channel))
    (words : List (Word (LocalLetter Lifeline Channel))) :
    words.foldl (· ++ ·) acc =
      acc ++ concatWords words := by
  induction words generalizing acc with
  | nil =>
      simp [concatWords]
  | cons word words ih =>
      simp only [List.foldl_cons]
      rw [ih (acc ++ word)]
      simpa [concatWords, List.append_assoc] using (ih word).symm

@[simp]
theorem concatWords_nil {Lifeline : Type u} {Channel : Type v} :
    concatWords ([] : List (Word (LocalLetter Lifeline Channel))) = [] :=
  rfl

@[simp]
theorem concatWords_cons {Lifeline : Type u} {Channel : Type v}
    (word : Word (LocalLetter Lifeline Channel))
    (words : List (Word (LocalLetter Lifeline Channel))) :
    concatWords (word :: words) =
      word ++ concatWords words := by
  simpa [concatWords] using
    (concatWords_foldl (Lifeline := Lifeline) (Channel := Channel) word words)

theorem concatWords_ofFn_congr {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    (left right : Fin n → Word (LocalLetter Lifeline Channel))
    (hwords : ∀ i, left i = right i) :
    concatWords (List.ofFn left) = concatWords (List.ofFn right) := by
  induction n with
  | zero =>
      simp
  | succ n ih =>
      simp [List.ofFn_succ, concatWords_cons, hwords 0,
        ih (fun i : Fin n => left i.succ) (fun i : Fin n => right i.succ)
          (fun i => hwords i.succ)]

/-- Complete local trace semantics for local programs. -/
inductive Trace {Lifeline : Type u} {Channel : Type v} :
    LocProg Lifeline Channel →
      Word (LocalLetter Lifeline Channel) → Prop where
  | eps :
      Trace LocProg.eps []
  | letter (letter : LocalLetter Lifeline Channel) :
      Trace (LocProg.letter letter) [letter]
  | seq {first second : LocProg Lifeline Channel}
      {firstWord secondWord : Word (LocalLetter Lifeline Channel)}
      (hfirst : Trace first firstWord)
      (hsecond : Trace second secondWord) :
      Trace (LocProg.seq first second) (firstWord ++ secondWord)
  | branch_true
      {trueLetter falseLetter : LocalLetter Lifeline Channel}
      {thenBranch elseBranch : LocProg Lifeline Channel}
      {rest : Word (LocalLetter Lifeline Channel)}
      (hrest : Trace thenBranch rest) :
      Trace
        (LocProg.branch trueLetter falseLetter thenBranch elseBranch)
        (trueLetter :: rest)
  | branch_false
      {trueLetter falseLetter : LocalLetter Lifeline Channel}
      {thenBranch elseBranch : LocProg Lifeline Channel}
      {rest : Word (LocalLetter Lifeline Channel)}
      (hrest : Trace elseBranch rest) :
      Trace
        (LocProg.branch trueLetter falseLetter thenBranch elseBranch)
        (falseLetter :: rest)
  | loop
      {trueLetter falseLetter : LocalLetter Lifeline Channel}
      {body exit : LocProg Lifeline Channel}
      {k : Nat}
      {bodies : Fin k → Word (LocalLetter Lifeline Channel)}
      {exitWord : Word (LocalLetter Lifeline Channel)}
      (hbodies : ∀ i, Trace body (bodies i))
      (hexit : Trace exit exitWord) :
      Trace
        (LocProg.loop trueLetter falseLetter body exit)
        (concatWords
          (List.ofFn (fun i : Fin k => trueLetter :: bodies i)) ++
          (falseLetter :: exitWord))
  | par {n : Nat}
      {branches : Fin n → LocProg Lifeline Channel}
      {word : Word (LocalLetter Lifeline Channel)}
      {tagged : Word (Tagged (Fin n) (LocalLetter Lifeline Channel))}
      {branchWords : Fin n → Word (LocalLetter Lifeline Channel)}
      (htagged : tagged.map Tagged.val = word)
      (herase : ∀ i, eraseBranch i tagged = branchWords i)
      (hbranches : ∀ i, Trace (branches i) (branchWords i)) :
      Trace (LocProg.par branches) word

abbrev trace {Lifeline : Type u} {Channel : Type v}
    (program : LocProg Lifeline Channel)
    (word : Word (LocalLetter Lifeline Channel)) : Prop :=
  Trace program word

/-- Prefix-freeness for a local word language. -/
def WordPrefixFree {α : Type u} (language : Word α → Prop) : Prop :=
  ∀ {left right}, language left → language right → ListPrefix left right →
    left = right

/--
Tagged complete trace of a local parallel program.

This keeps the branch-origin witness that is erased by `trace (par branches)`.
It is the form used for prefix-freeness in the parallel proof.
-/
def taggedParTrace {n : Nat} {Lifeline : Type u} {Channel : Type v}
    (branches : Fin n → LocProg Lifeline Channel)
    (tagged : Word (Tagged (Fin n) (LocalLetter Lifeline Channel))) : Prop :=
  ∃ branchWords : Fin n → Word (LocalLetter Lifeline Channel),
    (∀ i, eraseBranch i tagged = branchWords i) ∧
    ∀ i, trace (branches i) (branchWords i)

theorem taggedParTrace_to_trace {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {branches : Fin n → LocProg Lifeline Channel}
    {tagged : Word (Tagged (Fin n) (LocalLetter Lifeline Channel))}
    (h : taggedParTrace branches tagged) :
    trace (LocProg.par branches) (tagged.map Tagged.val) := by
  rcases h with ⟨branchWords, herase, hbranches⟩
  exact Trace.par rfl herase hbranches

theorem taggedParTrace_prefix_free {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {branches : Fin n → LocProg Lifeline Channel}
    (hprefixFree :
      ∀ i, WordPrefixFree (fun word => trace (branches i) word))
    {left right : Word (Tagged (Fin n) (LocalLetter Lifeline Channel))}
    (hleft : taggedParTrace branches left)
    (hright : taggedParTrace branches right)
    (hp : ListPrefix left right) :
    left = right := by
  rcases hleft with ⟨leftBranches, hleftErase, hleftTrace⟩
  rcases hright with ⟨rightBranches, hrightErase, hrightTrace⟩
  apply listPrefix_eq_of_eraseBranch_eq hp
  intro i
  have hprefix_i : ListPrefix (leftBranches i) (rightBranches i) := by
    have hpErase := eraseBranch_preserves_prefix (i := i) hp
    simpa [hleftErase i, hrightErase i] using hpErase
  have hbranch :
      leftBranches i = rightBranches i :=
    hprefixFree i (hleftTrace i) (hrightTrace i) hprefix_i
  calc
    eraseBranch i left = leftBranches i := hleftErase i
    _ = rightBranches i := hbranch
    _ = eraseBranch i right := (hrightErase i).symm

/-- Prefix closure of complete local traces. -/
def prefixTrace {Lifeline : Type u} {Channel : Type v}
    (program : LocProg Lifeline Channel)
    (word : Word (LocalLetter Lifeline Channel)) : Prop :=
  ∃ complete,
    trace program complete ∧ ListPrefix word complete

theorem trace_prefixTrace {Lifeline : Type u} {Channel : Type v}
    {program : LocProg Lifeline Channel}
    {word : Word (LocalLetter Lifeline Channel)}
    (h : trace program word) :
    prefixTrace program word :=
  ⟨word, h, ListPrefix.refl word⟩

theorem trace_eps {Lifeline : Type u} {Channel : Type v}
    (word : Word (LocalLetter Lifeline Channel)) :
    trace (LocProg.eps : LocProg Lifeline Channel) word ↔ word = [] := by
  constructor
  · intro h
    cases h
    rfl
  · intro h
    subst h
    exact Trace.eps

theorem trace_letter {Lifeline : Type u} {Channel : Type v}
    (letter : LocalLetter Lifeline Channel)
    (word : Word (LocalLetter Lifeline Channel)) :
    trace (LocProg.letter letter) word ↔ word = [letter] := by
  constructor
  · intro h
    cases h
    rfl
  · intro h
    subst h
    exact Trace.letter letter

theorem trace_seq {Lifeline : Type u} {Channel : Type v}
    (first second : LocProg Lifeline Channel)
    (word : Word (LocalLetter Lifeline Channel)) :
    trace (first ;;ₗ second) word ↔
      ∃ firstWord secondWord,
        trace first firstWord ∧
        trace second secondWord ∧
        word = firstWord ++ secondWord := by
  constructor
  · intro h
    cases h with
    | seq hfirst hsecond =>
        exact ⟨_, _, hfirst, hsecond, rfl⟩
  · rintro ⟨firstWord, secondWord, hfirst, hsecond, rfl⟩
    exact Trace.seq hfirst hsecond

theorem trace_branch {Lifeline : Type u} {Channel : Type v}
    (trueLetter falseLetter : LocalLetter Lifeline Channel)
    (thenBranch elseBranch : LocProg Lifeline Channel)
    (word : Word (LocalLetter Lifeline Channel)) :
    trace (LocProg.branch trueLetter falseLetter thenBranch elseBranch) word ↔
      (∃ rest,
        trace thenBranch rest ∧
        word = trueLetter :: rest) ∨
      (∃ rest,
        trace elseBranch rest ∧
        word = falseLetter :: rest) := by
  constructor
  · intro h
    cases h with
    | branch_true hrest =>
        exact Or.inl ⟨_, hrest, rfl⟩
    | branch_false hrest =>
        exact Or.inr ⟨_, hrest, rfl⟩
  · intro h
    rcases h with ⟨rest, hrest, rfl⟩ | ⟨rest, hrest, rfl⟩
    · exact Trace.branch_true hrest
    · exact Trace.branch_false hrest

theorem prefix_free_eps {Lifeline : Type u} {Channel : Type v} :
    WordPrefixFree
      (fun word =>
        trace (LocProg.eps : LocProg Lifeline Channel) word) := by
  intro left right hleft hright _hp
  rw [(trace_eps left).1 hleft, (trace_eps right).1 hright]

theorem prefix_free_letter {Lifeline : Type u} {Channel : Type v}
    (letter : LocalLetter Lifeline Channel) :
    WordPrefixFree (fun word => trace (LocProg.letter letter) word) := by
  intro left right hleft hright _hp
  rw [(trace_letter letter left).1 hleft,
    (trace_letter letter right).1 hright]

theorem prefix_free_seq {Lifeline : Type u} {Channel : Type v}
    {first second : LocProg Lifeline Channel}
    (hfirst : WordPrefixFree (fun word => trace first word))
    (hsecond : WordPrefixFree (fun word => trace second word)) :
    WordPrefixFree (fun word => trace (first ;;ₗ second) word) := by
  intro left right hleft hright hp
  rcases (trace_seq first second left).1 hleft with
    ⟨leftFirst, leftSecond, hleftFirst, hleftSecond, rfl⟩
  rcases (trace_seq first second right).1 hright with
    ⟨rightFirst, rightSecond, hrightFirst, hrightSecond, rfl⟩
  rcases hp with ⟨suffix, hsuffix⟩
  cases Nat.le_total leftFirst.length rightFirst.length with
  | inl hLen =>
      have hprefixFirst :
          ListPrefix leftFirst rightFirst :=
        ListPrefix.append_eq_left_prefix
          (u1 := leftFirst)
          (u2 := leftSecond ++ suffix)
          (v1 := rightFirst)
          (v2 := rightSecond)
          (by simpa [List.append_assoc] using hsuffix)
          hLen
      have hEqFirst : leftFirst = rightFirst :=
        hfirst hleftFirst hrightFirst hprefixFirst
      have hprefixSecond :
          ListPrefix leftSecond rightSecond := by
        refine ⟨suffix, ?_⟩
        have hcancel :
            leftFirst ++ (leftSecond ++ suffix) =
              leftFirst ++ rightSecond := by
          simpa [hEqFirst, List.append_assoc] using hsuffix
        exact List.append_cancel_left hcancel
      have hEqSecond : leftSecond = rightSecond :=
        hsecond hleftSecond hrightSecond hprefixSecond
      simp [hEqFirst, hEqSecond]
  | inr hLen =>
      have hprefixFirst :
          ListPrefix rightFirst leftFirst :=
        ListPrefix.append_eq_right_prefix
          (u1 := leftFirst)
          (u2 := leftSecond ++ suffix)
          (v1 := rightFirst)
          (v2 := rightSecond)
          (by simpa [List.append_assoc] using hsuffix)
          hLen
      have hEqFirst : rightFirst = leftFirst :=
        hfirst hrightFirst hleftFirst hprefixFirst
      have hprefixSecond :
          ListPrefix leftSecond rightSecond := by
        refine ⟨suffix, ?_⟩
        have hcancel :
            leftFirst ++ (leftSecond ++ suffix) =
              leftFirst ++ rightSecond := by
          simpa [hEqFirst, List.append_assoc] using hsuffix
        exact List.append_cancel_left hcancel
      have hEqSecond : leftSecond = rightSecond :=
        hsecond hleftSecond hrightSecond hprefixSecond
      simp [hEqFirst, hEqSecond]

theorem prefix_free_branch {Lifeline : Type u} {Channel : Type v}
    {trueLetter falseLetter : LocalLetter Lifeline Channel}
    {thenBranch elseBranch : LocProg Lifeline Channel}
    (hneq : trueLetter ≠ falseLetter)
    (hthen : WordPrefixFree (fun word => trace thenBranch word))
    (helse : WordPrefixFree (fun word => trace elseBranch word)) :
    WordPrefixFree
      (fun word =>
        trace
          (LocProg.branch trueLetter falseLetter thenBranch elseBranch)
          word) := by
  intro left right hleft hright hp
  rcases (trace_branch trueLetter falseLetter thenBranch elseBranch left).1
      hleft with
    ⟨leftRest, hleftRest, rfl⟩ | ⟨leftRest, hleftRest, rfl⟩
  · rcases (trace_branch trueLetter falseLetter thenBranch elseBranch right).1
      hright with
      ⟨rightRest, hrightRest, rfl⟩ | ⟨rightRest, _hrightRest, rfl⟩
    · have hpRest := ListPrefix.cons_cancel hp
      have hrestEq := hthen hleftRest hrightRest hpRest
      simp [hrestEq]
    · rcases hp with ⟨suffix, hsuffix⟩
      simp only [List.cons_append] at hsuffix
      have hhead : trueLetter = falseLetter := by
        cases hsuffix
        rfl
      exact False.elim (hneq hhead)
  · rcases (trace_branch trueLetter falseLetter thenBranch elseBranch right).1
      hright with
      ⟨rightRest, _hrightRest, rfl⟩ | ⟨rightRest, hrightRest, rfl⟩
    · rcases hp with ⟨suffix, hsuffix⟩
      simp only [List.cons_append] at hsuffix
      have hhead : falseLetter = trueLetter := by
        cases hsuffix
        rfl
      exact False.elim (hneq hhead.symm)
    · have hpRest := ListPrefix.cons_cancel hp
      have hrestEq := helse hleftRest hrightRest hpRest
      simp [hrestEq]

theorem trace_loop_unfold {Lifeline : Type u} {Channel : Type v}
    (trueLetter falseLetter : LocalLetter Lifeline Channel)
    (body exit : LocProg Lifeline Channel)
    (word : Word (LocalLetter Lifeline Channel)) :
    trace (LocProg.loop trueLetter falseLetter body exit) word ↔
      (∃ exitWord,
        trace exit exitWord ∧
        word = falseLetter :: exitWord) ∨
      (∃ bodyWord rest,
        trace body bodyWord ∧
        trace (LocProg.loop trueLetter falseLetter body exit) rest ∧
        word = trueLetter :: bodyWord ++ rest) := by
  constructor
  · intro h
    cases h with
    | loop hbodies hexit =>
        rename_i k bodies exitWord
        cases k with
        | zero =>
            left
            exact ⟨exitWord, hexit, by simp⟩
        | succ k =>
            right
            let bodiesTail :
                Fin k → Word (LocalLetter Lifeline Channel) :=
              fun i => bodies i.succ
            let rest :=
              concatWords
                (List.ofFn (fun i : Fin k => trueLetter :: bodiesTail i)) ++
              (falseLetter :: exitWord)
            refine ⟨bodies 0, rest, hbodies 0, ?_, ?_⟩
            · exact Trace.loop (fun i => hbodies i.succ) hexit
            · simp [rest, bodiesTail, List.ofFn_succ, List.append_assoc]
  · intro h
    rcases h with ⟨exitWord, hexit, rfl⟩ |
      ⟨bodyWord, rest, hbody, hrest, rfl⟩
    · simpa using
        (Trace.loop
          (trueLetter := trueLetter)
          (falseLetter := falseLetter)
          (body := body)
          (exit := exit)
          (k := 0)
          (bodies := Fin.elim0)
          (fun i => Fin.elim0 i)
          hexit)
    · cases hrest with
      | loop hbodies hexit =>
          rename_i k bodies exitWord
          let bodies' :
              Fin (k + 1) → Word (LocalLetter Lifeline Channel) :=
            Fin.cases bodyWord bodies
          have hbodies' :
              ∀ i, trace body (bodies' i) :=
            Fin.cases hbody hbodies
          simpa [bodies', List.ofFn_succ, List.append_assoc] using
            (Trace.loop
              (trueLetter := trueLetter)
              (falseLetter := falseLetter)
              (body := body)
              (exit := exit)
              (k := k + 1)
              (bodies := bodies')
              hbodies'
              hexit)

theorem prefix_free_loop {Lifeline : Type u} {Channel : Type v}
    {trueLetter falseLetter : LocalLetter Lifeline Channel}
    {body exit : LocProg Lifeline Channel}
    (hneq : trueLetter ≠ falseLetter)
    (hbody : WordPrefixFree (fun word => trace body word))
    (hexit : WordPrefixFree (fun word => trace exit word)) :
    WordPrefixFree
      (fun word =>
        trace (LocProg.loop trueLetter falseLetter body exit) word) := by
  let loopProgram := LocProg.loop trueLetter falseLetter body exit
  have hmain :
      ∀ m, ∀ left,
        left.length = m →
        trace loopProgram left →
        ∀ right,
          trace loopProgram right →
          ListPrefix left right →
          left = right := by
    intro m
    induction m using Nat.strongRecOn with
    | ind m ih =>
        intro left hlen hleft right hright hp
        rcases
            (trace_loop_unfold trueLetter falseLetter body exit left).1
              hleft with hleftExit | hleftIter
        ·
            rcases hleftExit with ⟨leftExit, hleftExit, rfl⟩
            rcases
                (trace_loop_unfold trueLetter falseLetter body exit right).1
                  hright with hrightExit | hrightIter
            ·
                rcases hrightExit with ⟨rightExit, hrightExit, rfl⟩
                have hpExit := ListPrefix.cons_cancel hp
                have hexitEq := hexit hleftExit hrightExit hpExit
                simp [hexitEq]
            ·
                rcases hrightIter with
                  ⟨rightBody, rightRest, _hrightBody, _hrightRest, rfl⟩
                rcases hp with ⟨suffix, hsuffix⟩
                simp only [List.cons_append] at hsuffix
                have hhead : falseLetter = trueLetter := by
                  injection hsuffix with hhead _htail
                exact False.elim (hneq hhead.symm)
        ·
            rcases hleftIter with
              ⟨leftBody, leftRest, hleftBody, hleftRest, rfl⟩
            rcases
                (trace_loop_unfold trueLetter falseLetter body exit right).1
                  hright with hrightExit | hrightIter
            ·
                rcases hrightExit with ⟨rightExit, _hrightExit, rfl⟩
                rcases hp with ⟨suffix, hsuffix⟩
                simp only [List.cons_append] at hsuffix
                have hhead : trueLetter = falseLetter := by
                  injection hsuffix with hhead _htail
                exact False.elim (hneq hhead)
            ·
                rcases hrightIter with
                  ⟨rightBody, rightRest, hrightBody, hrightRest, rfl⟩
                have hpTail : ListPrefix (leftBody ++ leftRest)
                    (rightBody ++ rightRest) :=
                  ListPrefix.cons_cancel hp
                rcases hpTail with ⟨suffix, hsuffix⟩
                have hlt : leftRest.length < m := by
                  rw [← hlen]
                  have hpos : 0 < leftBody.length + 1 :=
                    Nat.zero_lt_succ _
                  have hltBase :
                      leftRest.length <
                        (leftBody.length + 1) + leftRest.length :=
                    Nat.lt_add_of_pos_left (n := leftRest.length) hpos
                  simpa [List.length_append, Nat.add_assoc, Nat.add_comm,
                    Nat.add_left_comm] using hltBase
                cases Nat.le_total leftBody.length rightBody.length with
                | inl hLen =>
                    have hprefixBody : ListPrefix leftBody rightBody :=
                      ListPrefix.append_eq_left_prefix
                        (u1 := leftBody)
                        (u2 := leftRest ++ suffix)
                        (v1 := rightBody)
                        (v2 := rightRest)
                        (by simpa [List.append_assoc] using hsuffix)
                        hLen
                    have hEqBody : leftBody = rightBody :=
                      hbody hleftBody hrightBody hprefixBody
                    have hprefixRest : ListPrefix leftRest rightRest := by
                      refine ⟨suffix, ?_⟩
                      have hcancel :
                          leftBody ++ (leftRest ++ suffix) =
                            leftBody ++ rightRest := by
                        simpa [hEqBody, List.append_assoc] using hsuffix
                      exact List.append_cancel_left hcancel
                    have hEqRest : leftRest = rightRest :=
                      ih leftRest.length hlt leftRest rfl hleftRest
                        rightRest hrightRest hprefixRest
                    simp [hEqBody, hEqRest]
                | inr hLen =>
                    have hprefixBody : ListPrefix rightBody leftBody :=
                      ListPrefix.append_eq_right_prefix
                        (u1 := leftBody)
                        (u2 := leftRest ++ suffix)
                        (v1 := rightBody)
                        (v2 := rightRest)
                        (by simpa [List.append_assoc] using hsuffix)
                        hLen
                    have hEqBody : rightBody = leftBody :=
                      hbody hrightBody hleftBody hprefixBody
                    have hprefixRest : ListPrefix leftRest rightRest := by
                      refine ⟨suffix, ?_⟩
                      have hcancel :
                          leftBody ++ (leftRest ++ suffix) =
                            leftBody ++ rightRest := by
                        simpa [hEqBody, List.append_assoc] using hsuffix
                      exact List.append_cancel_left hcancel
                    have hEqRest : leftRest = rightRest :=
                      ih leftRest.length hlt leftRest rfl hleftRest
                        rightRest hrightRest hprefixRest
                    simp [hEqBody, hEqRest]
  intro left right hleft hright hp
  exact hmain left.length left rfl hleft right hright hp

/--
The guard letters emitted for true and false branches are distinct at every
control construct.

The concrete projection layer will discharge this for projected programs
because the two choice/control letters carry different Boolean values.
-/
def GuardDistinct {Lifeline : Type u} {Channel : Type v} :
    LocProg Lifeline Channel → Prop
  | LocProg.eps => True
  | LocProg.letter _ => True
  | LocProg.seq first second =>
      GuardDistinct first ∧ GuardDistinct second
  | LocProg.branch trueLetter falseLetter thenBranch elseBranch =>
      trueLetter ≠ falseLetter ∧
      GuardDistinct thenBranch ∧
      GuardDistinct elseBranch
  | LocProg.loop trueLetter falseLetter body exit =>
      trueLetter ≠ falseLetter ∧
      GuardDistinct body ∧
      GuardDistinct exit
  | LocProg.par branches =>
      ∀ i, GuardDistinct (branches i)

namespace GuardDistinct

theorem seq {Lifeline : Type u} {Channel : Type v}
    {first second : LocProg Lifeline Channel}
    (hfirst : GuardDistinct first)
    (hsecond : GuardDistinct second) :
    GuardDistinct (LocProg.seq first second) :=
  ⟨hfirst, hsecond⟩

theorem par {n : Nat} {Lifeline : Type u} {Channel : Type v}
    {branches : Fin n → LocProg Lifeline Channel}
    (hbranches : ∀ i, GuardDistinct (branches i)) :
    GuardDistinct (LocProg.par branches) :=
  hbranches

theorem branch_choice_if {Lifeline : Type u} {Channel : Type v}
    (tag : ControlTag)
    {thenBranch elseBranch : LocProg Lifeline Channel}
    (hthen : GuardDistinct thenBranch)
    (helse : GuardDistinct elseBranch) :
    GuardDistinct
      (LocProg.branch
        (LocalLetter.choice tag ChoiceKind.ifTrue)
        (LocalLetter.choice tag ChoiceKind.ifFalse)
        thenBranch elseBranch) :=
  ⟨LocalLetter.choice_ifTrue_ne_ifFalse tag, hthen, helse⟩

theorem loop_choice_while {Lifeline : Type u} {Channel : Type v}
    (tag : ControlTag)
    {body exit : LocProg Lifeline Channel}
    (hbody : GuardDistinct body)
    (hexit : GuardDistinct exit) :
    GuardDistinct
      (LocProg.loop
        (LocalLetter.choice tag ChoiceKind.whileTrue)
        (LocalLetter.choice tag ChoiceKind.whileFalse)
        body exit) :=
  ⟨LocalLetter.choice_whileTrue_ne_whileFalse tag, hbody, hexit⟩

theorem branch_ctrlSend_if {Lifeline : Type u} {Channel : Type v}
    (B : Lifeline) (ch : Channel) (tag : ControlTag)
    {thenBranch elseBranch : LocProg Lifeline Channel}
    (hthen : GuardDistinct thenBranch)
    (helse : GuardDistinct elseBranch) :
    GuardDistinct
      (LocProg.branch
        (LocalLetter.ctrlSend B ch tag ChoiceKind.ifTrue)
        (LocalLetter.ctrlSend B ch tag ChoiceKind.ifFalse)
        thenBranch elseBranch) :=
  ⟨LocalLetter.ctrlSend_ifTrue_ne_ifFalse B ch tag, hthen, helse⟩

theorem branch_ctrlRecv_if {Lifeline : Type u} {Channel : Type v}
    (A : Lifeline) (ch : Channel) (tag : ControlTag)
    {thenBranch elseBranch : LocProg Lifeline Channel}
    (hthen : GuardDistinct thenBranch)
    (helse : GuardDistinct elseBranch) :
    GuardDistinct
      (LocProg.branch
        (LocalLetter.ctrlRecv A ch tag ChoiceKind.ifTrue)
        (LocalLetter.ctrlRecv A ch tag ChoiceKind.ifFalse)
        thenBranch elseBranch) :=
  ⟨LocalLetter.ctrlRecv_ifTrue_ne_ifFalse A ch tag, hthen, helse⟩

theorem loop_ctrlSend_while {Lifeline : Type u} {Channel : Type v}
    (B : Lifeline) (ch : Channel) (tag : ControlTag)
    {body exit : LocProg Lifeline Channel}
    (hbody : GuardDistinct body)
    (hexit : GuardDistinct exit) :
    GuardDistinct
      (LocProg.loop
        (LocalLetter.ctrlSend B ch tag ChoiceKind.whileTrue)
        (LocalLetter.ctrlSend B ch tag ChoiceKind.whileFalse)
        body exit) :=
  ⟨LocalLetter.ctrlSend_whileTrue_ne_whileFalse B ch tag, hbody, hexit⟩

theorem loop_ctrlRecv_while {Lifeline : Type u} {Channel : Type v}
    (A : Lifeline) (ch : Channel) (tag : ControlTag)
    {body exit : LocProg Lifeline Channel}
    (hbody : GuardDistinct body)
    (hexit : GuardDistinct exit) :
    GuardDistinct
      (LocProg.loop
        (LocalLetter.ctrlRecv A ch tag ChoiceKind.whileTrue)
        (LocalLetter.ctrlRecv A ch tag ChoiceKind.whileFalse)
        body exit) :=
  ⟨LocalLetter.ctrlRecv_whileTrue_ne_whileFalse A ch tag, hbody, hexit⟩

end GuardDistinct

/-- The ISoLA core fragment, excluding local parallel interleaving. -/
def NoPar {Lifeline : Type u} {Channel : Type v} :
    LocProg Lifeline Channel → Prop
  | LocProg.eps => True
  | LocProg.letter _ => True
  | LocProg.seq first second =>
      NoPar first ∧ NoPar second
  | LocProg.branch _ _ thenBranch elseBranch =>
      NoPar thenBranch ∧ NoPar elseBranch
  | LocProg.loop _ _ body exit =>
      NoPar body ∧ NoPar exit
  | LocProg.par _ => False

theorem prefix_free_noPar {Lifeline : Type u} {Channel : Type v}
    (program : LocProg Lifeline Channel)
    (hguard : GuardDistinct program)
    (hnopar : NoPar program) :
    WordPrefixFree (fun word => trace program word) := by
  induction program with
  | eps =>
      exact prefix_free_eps
  | letter letter =>
      exact prefix_free_letter letter
  | seq first second ihFirst ihSecond =>
      rcases hguard with ⟨hguardFirst, hguardSecond⟩
      rcases hnopar with ⟨hnoparFirst, hnoparSecond⟩
      exact prefix_free_seq
        (ihFirst hguardFirst hnoparFirst)
        (ihSecond hguardSecond hnoparSecond)
  | branch trueLetter falseLetter thenBranch elseBranch ihThen ihElse =>
      rcases hguard with
        ⟨hneq, hguardThen, hguardElse⟩
      rcases hnopar with ⟨hnoparThen, hnoparElse⟩
      exact prefix_free_branch hneq
        (ihThen hguardThen hnoparThen)
        (ihElse hguardElse hnoparElse)
  | loop trueLetter falseLetter body exit ihBody ihExit =>
      rcases hguard with
        ⟨hneq, hguardBody, hguardExit⟩
      rcases hnopar with ⟨hnoparBody, hnoparExit⟩
      exact prefix_free_loop hneq
        (ihBody hguardBody hnoparBody)
        (ihExit hguardExit hnoparExit)
  | par branches =>
      exact False.elim hnopar

theorem localPrefixFree_noPar {Lifeline : Type u} {Channel : Type v}
    {program : Lifeline → LocProg Lifeline Channel}
    (hguard : ∀ A, GuardDistinct (program A))
    (hnopar : ∀ A, NoPar (program A)) :
    LocalPrefixFree (fun A word => trace (program A) word) := by
  intro A xs ys hxs hys hp
  exact prefix_free_noPar (program A) (hguard A) (hnopar A) hxs hys hp

theorem componentPrefixFree_noPar {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    (hguard : ∀ i A, GuardDistinct (program i A))
    (hnopar : ∀ i A, NoPar (program i A)) :
    ComponentPrefixFree
      (fun i A word => trace (program i A) word) := by
  intro i A xs ys hxs hys hp
  exact prefix_free_noPar (program i A) (hguard i A) (hnopar i A)
    hxs hys hp

theorem taggedParTrace_prefix_free_noPar {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {branches : Fin n → LocProg Lifeline Channel}
    (hguard : ∀ i, GuardDistinct (branches i))
    (hnopar : ∀ i, NoPar (branches i)) :
    WordPrefixFree
      (fun tagged =>
        taggedParTrace branches tagged) :=
  taggedParTrace_prefix_free
    (fun i => prefix_free_noPar (branches i) (hguard i) (hnopar i))

theorem taggedParLocCompleteLocal_prefix_free_noPar {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    (hguard : ∀ i A, GuardDistinct (program i A))
    (hnopar : ∀ i A, NoPar (program i A)) :
    LocalPrefixFree
      (TaggedParLocCompleteLocal
        (fun i A word => trace (program i A) word)) :=
  TaggedParLocCompleteLocal.prefix_free
    (componentPrefixFree_noPar hguard hnopar)

theorem taggedParLocComplete_prefix_free_noPar {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    (hguard : ∀ i A, GuardDistinct (program i A))
    (hnopar : ∀ i A, NoPar (program i A))
    {u v : ParLocTaggedTrace n Lifeline Channel}
    (hu :
      TaggedParLocComplete
        (fun i A word => trace (program i A) word) u)
    (hv :
      TaggedParLocComplete
        (fun i A word => trace (program i A) word) v)
    (hp : TracePrefix u v) :
    u = v :=
  parloc_complete_prefix_free
    (componentPrefixFree_noPar hguard hnopar) hu hv hp

theorem trace_par_intro {n : Nat} {Lifeline : Type u} {Channel : Type v}
    {branches : Fin n → LocProg Lifeline Channel}
    {word : Word (LocalLetter Lifeline Channel)}
    {tagged : Word (Tagged (Fin n) (LocalLetter Lifeline Channel))}
    {branchWords : Fin n → Word (LocalLetter Lifeline Channel)}
    (htagged : tagged.map Tagged.val = word)
    (herase : ∀ i, eraseBranch i tagged = branchWords i)
    (hbranches : ∀ i, trace (branches i) (branchWords i)) :
    trace (LocProg.par branches) word := by
  exact Trace.par htagged herase hbranches

theorem trace_par_elim {n : Nat} {Lifeline : Type u} {Channel : Type v}
    {branches : Fin n → LocProg Lifeline Channel}
    {word : Word (LocalLetter Lifeline Channel)}
    (h : trace (LocProg.par branches) word) :
    ∃ tagged : Word (Tagged (Fin n) (LocalLetter Lifeline Channel)),
      tagged.map Tagged.val = word ∧
      ∃ branchWords : Fin n → Word (LocalLetter Lifeline Channel),
        (∀ i, eraseBranch i tagged = branchWords i) ∧
        ∀ i, trace (branches i) (branchWords i) :=
by
  cases h with
  | par htagged herase hbranches =>
      exact ⟨_, htagged, _, herase, hbranches⟩

theorem trace_par {n : Nat} {Lifeline : Type u} {Channel : Type v}
    (branches : Fin n → LocProg Lifeline Channel)
    (word : Word (LocalLetter Lifeline Channel)) :
    trace (LocProg.par branches) word ↔
      ∃ tagged : Word (Tagged (Fin n) (LocalLetter Lifeline Channel)),
        tagged.map Tagged.val = word ∧
        ∃ branchWords : Fin n → Word (LocalLetter Lifeline Channel),
          (∀ i, eraseBranch i tagged = branchWords i) ∧
          ∀ i, trace (branches i) (branchWords i) := by
  constructor
  · exact trace_par_elim
  · rintro ⟨tagged, htagged, branchWords, herase, hbranches⟩
    exact trace_par_intro htagged herase hbranches

theorem taggedParLocCompleteLocal_to_trace_par {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {A : Lifeline}
    {word : Word (Tagged (Fin n) (LocalLetter Lifeline Channel))}
    (hcomplete :
      TaggedParLocCompleteLocal
        (fun i A word => trace (program i A) word) A word) :
    trace (LocProg.par (fun i => program i A)) (word.map Tagged.val) := by
  rcases hcomplete with ⟨branches, herase, hbranchesComplete⟩
  exact Trace.par rfl herase hbranchesComplete

inductive DecoratedLetter (Lifeline : Type u) (Channel : Type v) :
    Type (max u v) where
  | atom (letter : LocalLetter Lifeline Channel) :
      DecoratedLetter Lifeline Channel
  | branch {n : Nat} (i : Fin n) (letter : DecoratedLetter Lifeline Channel) :
      DecoratedLetter Lifeline Channel
deriving Repr

namespace DecoratedLetter

def toLocal {Lifeline : Type u} {Channel : Type v} :
    DecoratedLetter Lifeline Channel → LocalLetter Lifeline Channel
  | atom a => a
  | branch _ a => toLocal a

@[simp]
theorem toLocal_atom {Lifeline : Type u} {Channel : Type v}
    (letter : LocalLetter Lifeline Channel) :
    toLocal (atom letter) = letter :=
  rfl

@[simp]
theorem toLocal_branch {n : Nat} {Lifeline : Type u} {Channel : Type v}
    (i : Fin n) (letter : DecoratedLetter Lifeline Channel) :
    toLocal (branch i letter) = toLocal letter :=
  rfl

end DecoratedLetter

def wrapTaggedDecorated {n : Nat} {Lifeline : Type u} {Channel : Type v}
    (letter : Tagged (Fin n) (DecoratedLetter Lifeline Channel)) :
    DecoratedLetter Lifeline Channel :=
  DecoratedLetter.branch letter.branch letter.val

theorem wrapTaggedDecorated_injective {n : Nat}
    {Lifeline : Type u} {Channel : Type v} :
    Function.Injective
      (wrapTaggedDecorated
        (n := n) (Lifeline := Lifeline) (Channel := Channel)) := by
  intro x y h
  cases x with
  | mk xb xv =>
      cases y with
      | mk yb yv =>
          injection h with _ hbranch hval
          cases hbranch
          cases hval
          rfl

theorem listPrefix_map_reflects {α : Type u} {β : Type v}
    {f : α → β} (hinj : Function.Injective f)
    {xs ys : List α}
    (h : ListPrefix (xs.map f) (ys.map f)) :
    ListPrefix xs ys := by
  revert ys
  induction xs with
  | nil =>
      intro ys _h
      exact ⟨ys, rfl⟩
  | cons x xs ih =>
      intro ys h
      cases ys with
      | nil =>
          rcases h with ⟨suffix, hsuffix⟩
          simp at hsuffix
      | cons y ys =>
          rcases h with ⟨suffix, hsuffix⟩
          simp only [List.map_cons, List.cons_append] at hsuffix
          injection hsuffix with hhead htail
          have hxy : x = y := hinj hhead
          subst y
          rcases ih ⟨suffix, htail⟩ with ⟨rest, hrest⟩
          exact ⟨rest, by simp [hrest]⟩

theorem eraseBranch_map_decorated_toLocal {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    (i : Fin n)
    (xs : Word (Tagged (Fin n) (DecoratedLetter Lifeline Channel))) :
    eraseBranch i
        (xs.map
          (fun x =>
            ({ branch := x.branch, val := DecoratedLetter.toLocal x.val } :
              Tagged (Fin n) (LocalLetter Lifeline Channel)))) =
      (eraseBranch i xs).map DecoratedLetter.toLocal := by
  induction xs with
  | nil =>
      simp [eraseBranch]
  | cons x xs ih =>
      by_cases hx : x.branch = i
      · simp [eraseBranch, hx, ih]
      · simp [eraseBranch, hx, ih]

/--
Decorated complete local traces.

This is the recursive tagged trace language used for prefix-freeness.  It
records branch-origin wrappers at every nested `par`, while
`DecoratedLetter.toLocal` forgets those wrappers back to the ordinary local
alphabet.
-/
inductive DTrace {Lifeline : Type u} {Channel : Type v} :
    LocProg Lifeline Channel →
      Word (DecoratedLetter Lifeline Channel) → Prop where
  | eps :
      DTrace LocProg.eps []
  | letter (letter : LocalLetter Lifeline Channel) :
      DTrace (LocProg.letter letter) [DecoratedLetter.atom letter]
  | seq {first second : LocProg Lifeline Channel}
      {firstWord secondWord : Word (DecoratedLetter Lifeline Channel)}
      (hfirst : DTrace first firstWord)
      (hsecond : DTrace second secondWord) :
      DTrace (LocProg.seq first second) (firstWord ++ secondWord)
  | branch_true
      {trueLetter falseLetter : LocalLetter Lifeline Channel}
      {thenBranch elseBranch : LocProg Lifeline Channel}
      {rest : Word (DecoratedLetter Lifeline Channel)}
      (hrest : DTrace thenBranch rest) :
      DTrace
        (LocProg.branch trueLetter falseLetter thenBranch elseBranch)
        (DecoratedLetter.atom trueLetter :: rest)
  | branch_false
      {trueLetter falseLetter : LocalLetter Lifeline Channel}
      {thenBranch elseBranch : LocProg Lifeline Channel}
      {rest : Word (DecoratedLetter Lifeline Channel)}
      (hrest : DTrace elseBranch rest) :
      DTrace
        (LocProg.branch trueLetter falseLetter thenBranch elseBranch)
        (DecoratedLetter.atom falseLetter :: rest)
  | loop_exit
      {trueLetter falseLetter : LocalLetter Lifeline Channel}
      {body exit : LocProg Lifeline Channel}
      {exitWord : Word (DecoratedLetter Lifeline Channel)}
      (hexit : DTrace exit exitWord) :
      DTrace
        (LocProg.loop trueLetter falseLetter body exit)
        (DecoratedLetter.atom falseLetter :: exitWord)
  | loop_iter
      {trueLetter falseLetter : LocalLetter Lifeline Channel}
      {body exit : LocProg Lifeline Channel}
      {bodyWord rest : Word (DecoratedLetter Lifeline Channel)}
      (hbody : DTrace body bodyWord)
      (hrest : DTrace (LocProg.loop trueLetter falseLetter body exit) rest) :
      DTrace
        (LocProg.loop trueLetter falseLetter body exit)
        (DecoratedLetter.atom trueLetter :: bodyWord ++ rest)
  | par {n : Nat}
      {branches : Fin n → LocProg Lifeline Channel}
      {tagged : Word (Tagged (Fin n) (DecoratedLetter Lifeline Channel))}
      (hbranches : ∀ i, DTrace (branches i) (eraseBranch i tagged)) :
      DTrace (LocProg.par branches) (tagged.map wrapTaggedDecorated)

theorem dtrace_par_elim {n : Nat} {Lifeline : Type u} {Channel : Type v}
    {branches : Fin n → LocProg Lifeline Channel}
    {word : Word (DecoratedLetter Lifeline Channel)}
    (h : DTrace (LocProg.par branches) word) :
    ∃ tagged : Word (Tagged (Fin n) (DecoratedLetter Lifeline Channel)),
      word = tagged.map wrapTaggedDecorated ∧
      ∀ i, DTrace (branches i) (eraseBranch i tagged) := by
  cases h with
  | par hbranches =>
      exact ⟨_, rfl, hbranches⟩

theorem dtrace_par {n : Nat} {Lifeline : Type u} {Channel : Type v}
    (branches : Fin n → LocProg Lifeline Channel)
    (word : Word (DecoratedLetter Lifeline Channel)) :
    DTrace (LocProg.par branches) word ↔
      ∃ tagged : Word (Tagged (Fin n) (DecoratedLetter Lifeline Channel)),
        word = tagged.map wrapTaggedDecorated ∧
        ∀ i, DTrace (branches i) (eraseBranch i tagged) := by
  constructor
  · exact dtrace_par_elim
  · rintro ⟨tagged, rfl, hbranches⟩
    exact DTrace.par hbranches

def loopDecoratedTrace {Lifeline : Type u} {Channel : Type v}
    (trueLetter falseLetter : LocalLetter Lifeline Channel)
    (body exit : LocProg Lifeline Channel)
    (bodies : List {word : Word (DecoratedLetter Lifeline Channel) //
      DTrace body word})
    (exitWord : {word : Word (DecoratedLetter Lifeline Channel) //
      DTrace exit word}) :
    {word : Word (DecoratedLetter Lifeline Channel) //
      DTrace (LocProg.loop trueLetter falseLetter body exit) word} :=
  match bodies with
  | [] =>
      ⟨DecoratedLetter.atom falseLetter :: exitWord.val,
        DTrace.loop_exit exitWord.property⟩
  | bodyWord :: rest =>
      let restTrace :=
        loopDecoratedTrace trueLetter falseLetter body exit rest exitWord
      ⟨DecoratedLetter.atom trueLetter :: bodyWord.val ++ restTrace.val,
        DTrace.loop_iter bodyWord.property restTrace.property⟩

theorem loopDecoratedTrace_toLocal {Lifeline : Type u} {Channel : Type v}
    (trueLetter falseLetter : LocalLetter Lifeline Channel)
    (body exit : LocProg Lifeline Channel)
    (bodies : List {word : Word (DecoratedLetter Lifeline Channel) //
      DTrace body word})
    (exitWord : {word : Word (DecoratedLetter Lifeline Channel) //
      DTrace exit word}) :
    (loopDecoratedTrace trueLetter falseLetter body exit bodies exitWord).val.map
        DecoratedLetter.toLocal =
      concatWords
        (bodies.map
          (fun bodyWord => trueLetter ::
            bodyWord.val.map DecoratedLetter.toLocal)) ++
        (falseLetter :: exitWord.val.map DecoratedLetter.toLocal) := by
  induction bodies with
  | nil =>
      simp [loopDecoratedTrace, concatWords]
  | cons bodyWord rest ih =>
      simp [loopDecoratedTrace, concatWords_cons, List.map_append, ih]

theorem dtrace_par_complete_local_of_branches {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {branches : Fin n → LocProg Lifeline Channel}
    {tagged : Word (Tagged (Fin n) (DecoratedLetter Lifeline Channel))}
    (hbranches : ∀ i, DTrace (branches i) (eraseBranch i tagged))
    (A : Lifeline) :
    TaggedParLocCompleteLocalOf
      (fun i _ word => DTrace (branches i) word) A tagged :=
  ⟨fun i => eraseBranch i tagged, fun _ => rfl, hbranches⟩

theorem trace_to_dtrace {Lifeline : Type u} {Channel : Type v}
    {program : LocProg Lifeline Channel}
    {word : Word (LocalLetter Lifeline Channel)}
    (h : trace program word) :
    ∃ decorated : Word (DecoratedLetter Lifeline Channel),
      DTrace program decorated ∧
      decorated.map DecoratedLetter.toLocal = word := by
  classical
  induction h with
  | eps =>
      exact ⟨[], DTrace.eps, rfl⟩
  | letter letter =>
      exact ⟨[DecoratedLetter.atom letter], DTrace.letter letter, rfl⟩
  | seq hfirst hsecond ihFirst ihSecond =>
      rcases ihFirst with ⟨firstDecorated, hfirstDecorated, hfirstMap⟩
      rcases ihSecond with ⟨secondDecorated, hsecondDecorated, hsecondMap⟩
      refine ⟨firstDecorated ++ secondDecorated,
        DTrace.seq hfirstDecorated hsecondDecorated, ?_⟩
      simp [List.map_append, hfirstMap, hsecondMap]
  | branch_true hrest ihRest =>
      rcases ihRest with ⟨restDecorated, hrestDecorated, hrestMap⟩
      exact ⟨DecoratedLetter.atom _ :: restDecorated,
        DTrace.branch_true hrestDecorated, by simp [hrestMap]⟩
  | branch_false hrest ihRest =>
      rcases ihRest with ⟨restDecorated, hrestDecorated, hrestMap⟩
      exact ⟨DecoratedLetter.atom _ :: restDecorated,
        DTrace.branch_false hrestDecorated, by simp [hrestMap]⟩
  | loop hbodies hexit ihBodies ihExit =>
      rename_i trueLetter falseLetter body exit k bodies exitWord
      let bodyDecorated :
          Fin k → {word : Word (DecoratedLetter Lifeline Channel) //
            DTrace body word} :=
        fun i =>
          ⟨Classical.choose (ihBodies i),
            (Classical.choose_spec (ihBodies i)).1⟩
      let bodyDecoratedList :
          List {word : Word (DecoratedLetter Lifeline Channel) //
            DTrace body word} :=
        List.ofFn bodyDecorated
      let exitDecorated :
          {word : Word (DecoratedLetter Lifeline Channel) //
            DTrace exit word} :=
        ⟨Classical.choose ihExit, (Classical.choose_spec ihExit).1⟩
      refine
        ⟨(loopDecoratedTrace trueLetter falseLetter body exit
            bodyDecoratedList exitDecorated).val,
          (loopDecoratedTrace trueLetter falseLetter body exit
            bodyDecoratedList exitDecorated).property, ?_⟩
      have hbodyMap :
          concatWords
            (bodyDecoratedList.map
              (fun bodyWord => trueLetter ::
                bodyWord.val.map DecoratedLetter.toLocal)) =
            concatWords
              (List.ofFn (fun i : Fin k => trueLetter :: bodies i)) := by
        change concatWords
            ((List.ofFn bodyDecorated).map
              (fun bodyWord => trueLetter ::
                bodyWord.val.map DecoratedLetter.toLocal)) =
          concatWords
            (List.ofFn (fun i : Fin k => trueLetter :: bodies i))
        rw [List.map_ofFn]
        exact concatWords_ofFn_congr _ _
          (fun i => by
            simp [bodyDecorated, (Classical.choose_spec (ihBodies i)).2])
      rw [loopDecoratedTrace_toLocal, hbodyMap,
        (Classical.choose_spec ihExit).2]
  | par htagged herase hbranches ihBranches =>
      rename_i n branches word tagged branchWords
      let branchDecorated :
          Fin n → Word (DecoratedLetter Lifeline Channel) :=
        fun i => Classical.choose (ihBranches i)
      have hbranchDecorated :
          ∀ i, DTrace (branches i) (branchDecorated i) := by
        intro i
        exact (Classical.choose_spec (ihBranches i)).1
      have hbranchMap :
          ∀ i, (branchDecorated i).map DecoratedLetter.toLocal =
            eraseBranch i tagged := by
        intro i
        calc
          (branchDecorated i).map DecoratedLetter.toLocal = branchWords i :=
            (Classical.choose_spec (ihBranches i)).2
          _ = eraseBranch i tagged := (herase i).symm
      rcases taggedWord_lift_of_eraseBranch_map
          DecoratedLetter.toLocal tagged branchDecorated hbranchMap with
        ⟨taggedDecorated, htaggedDecoratedMap, htaggedDecoratedErase⟩
      refine ⟨taggedDecorated.map wrapTaggedDecorated,
        DTrace.par ?_, ?_⟩
      · intro i
        rw [htaggedDecoratedErase i]
        exact hbranchDecorated i
      · calc
          (taggedDecorated.map wrapTaggedDecorated).map
              DecoratedLetter.toLocal =
            (taggedDecorated.map
              (Tagged.mapVal DecoratedLetter.toLocal)).map Tagged.val := by
                simp [wrapTaggedDecorated, Tagged.mapVal, List.map_map,
                  Function.comp_def]
          _ = tagged.map Tagged.val := by rw [htaggedDecoratedMap]
          _ = word := htagged

theorem dtrace_to_trace {Lifeline : Type u} {Channel : Type v}
    {program : LocProg Lifeline Channel}
    {word : Word (DecoratedLetter Lifeline Channel)}
    (h : DTrace program word) :
    trace program (word.map DecoratedLetter.toLocal) := by
  induction h with
  | eps =>
      exact Trace.eps
  | letter letter =>
      exact Trace.letter letter
  | seq hfirst hsecond ihFirst ihSecond =>
      simpa [List.map_append] using Trace.seq ihFirst ihSecond
  | branch_true hrest ihRest =>
      exact Trace.branch_true ihRest
  | branch_false hrest ihRest =>
      exact Trace.branch_false ihRest
  | loop_exit hexit ihExit =>
      exact (trace_loop_unfold _ _ _ _ _).2
        (Or.inl ⟨_, ihExit, rfl⟩)
  | loop_iter hbody hrest ihBody ihRest =>
      exact (trace_loop_unfold _ _ _ _ _).2
        (Or.inr ⟨_, _, ihBody, ihRest, by simp [List.map_append]⟩)
  | par hbranches ihBranches =>
      rename_i n branches tagged
      let taggedLocal :
          Word (Tagged (Fin n) (LocalLetter Lifeline Channel)) :=
        tagged.map
          (fun x =>
            ({ branch := x.branch, val := DecoratedLetter.toLocal x.val } :
              Tagged (Fin n) (LocalLetter Lifeline Channel)))
      refine Trace.par
        (tagged := taggedLocal)
        (branchWords :=
          fun i => (eraseBranch i tagged).map DecoratedLetter.toLocal)
        ?_ ?_ ?_
      · simp [taggedLocal, wrapTaggedDecorated]
      · intro i
        exact eraseBranch_map_decorated_toLocal i tagged
      · intro i
        exact ihBranches i

theorem dtrace_prefix_free {Lifeline : Type u} {Channel : Type v}
    (program : LocProg Lifeline Channel)
    (hguard : GuardDistinct program) :
    WordPrefixFree (fun word => DTrace program word) := by
  induction program with
  | eps =>
      intro left right hleft hright _hp
      cases hleft
      cases hright
      rfl
  | letter letter =>
      intro left right hleft hright _hp
      cases hleft
      cases hright
      rfl
  | seq first second ihFirst ihSecond =>
      rcases hguard with ⟨hguardFirst, hguardSecond⟩
      intro left right hleft hright hp
      cases hleft with
      | seq hleftFirst hleftSecond =>
          rename_i leftFirst leftSecond
          cases hright with
          | seq hrightFirst hrightSecond =>
              rename_i rightFirst rightSecond
              rcases hp with ⟨suffix, hsuffix⟩
              cases Nat.le_total leftFirst.length rightFirst.length with
              | inl hLen =>
                  have hprefixFirst :
                      ListPrefix leftFirst rightFirst :=
                    ListPrefix.append_eq_left_prefix
                      (u1 := leftFirst)
                      (u2 := leftSecond ++ suffix)
                      (v1 := rightFirst)
                      (v2 := rightSecond)
                      (by simpa [List.append_assoc] using hsuffix)
                      hLen
                  have hEqFirst : leftFirst = rightFirst :=
                    ihFirst hguardFirst hleftFirst hrightFirst hprefixFirst
                  have hprefixSecond :
                      ListPrefix leftSecond rightSecond := by
                    refine ⟨suffix, ?_⟩
                    have hcancel :
                        leftFirst ++ (leftSecond ++ suffix) =
                          leftFirst ++ rightSecond := by
                      simpa [hEqFirst, List.append_assoc] using hsuffix
                    exact List.append_cancel_left hcancel
                  have hEqSecond : leftSecond = rightSecond :=
                    ihSecond hguardSecond hleftSecond hrightSecond hprefixSecond
                  simp [hEqFirst, hEqSecond]
              | inr hLen =>
                  have hprefixFirst :
                      ListPrefix rightFirst leftFirst :=
                    ListPrefix.append_eq_right_prefix
                      (u1 := leftFirst)
                      (u2 := leftSecond ++ suffix)
                      (v1 := rightFirst)
                      (v2 := rightSecond)
                      (by simpa [List.append_assoc] using hsuffix)
                      hLen
                  have hEqFirst : rightFirst = leftFirst :=
                    ihFirst hguardFirst hrightFirst hleftFirst hprefixFirst
                  have hprefixSecond :
                      ListPrefix leftSecond rightSecond := by
                    refine ⟨suffix, ?_⟩
                    have hcancel :
                        leftFirst ++ (leftSecond ++ suffix) =
                          leftFirst ++ rightSecond := by
                      simpa [hEqFirst, List.append_assoc] using hsuffix
                    exact List.append_cancel_left hcancel
                  have hEqSecond : leftSecond = rightSecond :=
                    ihSecond hguardSecond hleftSecond hrightSecond hprefixSecond
                  simp [hEqFirst, hEqSecond]
  | branch trueLetter falseLetter thenBranch elseBranch ihThen ihElse =>
      rcases hguard with ⟨hneq, hguardThen, hguardElse⟩
      intro left right hleft hright hp
      cases hleft with
      | branch_true hleftRest =>
          cases hright with
          | branch_true hrightRest =>
              have hpRest := ListPrefix.cons_cancel hp
              have hrestEq := ihThen hguardThen hleftRest hrightRest hpRest
              simp [hrestEq]
          | branch_false hrightRest =>
              rcases hp with ⟨suffix, hsuffix⟩
              simp only [List.cons_append] at hsuffix
              have hhead : trueLetter = falseLetter := by
                injection hsuffix with hhead _htail
                cases hhead
                rfl
              exact False.elim (hneq hhead)
      | branch_false hleftRest =>
          cases hright with
          | branch_true hrightRest =>
              rcases hp with ⟨suffix, hsuffix⟩
              simp only [List.cons_append] at hsuffix
              have hhead : falseLetter = trueLetter := by
                injection hsuffix with hhead _htail
                cases hhead
                rfl
              exact False.elim (hneq hhead.symm)
          | branch_false hrightRest =>
              have hpRest := ListPrefix.cons_cancel hp
              have hrestEq := ihElse hguardElse hleftRest hrightRest hpRest
              simp [hrestEq]
  | loop trueLetter falseLetter body exit ihBody ihExit =>
      rcases hguard with ⟨hneq, hguardBody, hguardExit⟩
      have hmain :
          ∀ m, ∀ left,
            left.length = m →
            DTrace (LocProg.loop trueLetter falseLetter body exit) left →
            ∀ right,
              DTrace (LocProg.loop trueLetter falseLetter body exit) right →
              ListPrefix left right →
              left = right := by
        intro m
        induction m using Nat.strongRecOn with
        | ind m ih =>
            intro left hlen hleft right hright hp
            cases hleft with
            | loop_exit hleftExit =>
                cases hright with
                | loop_exit hrightExit =>
                    have hpExit := ListPrefix.cons_cancel hp
                    have hEqExit :=
                      ihExit hguardExit hleftExit hrightExit hpExit
                    simp [hEqExit]
                | loop_iter hrightBody hrightRest =>
                    rcases hp with ⟨suffix, hsuffix⟩
                    simp only [List.cons_append] at hsuffix
                    have hhead : falseLetter = trueLetter := by
                      injection hsuffix with hhead _htail
                      cases hhead
                      rfl
                    exact False.elim (hneq hhead.symm)
            | loop_iter hleftBody hleftRest =>
                rename_i leftBody leftRest
                cases hright with
                | loop_exit hrightExit =>
                    rcases hp with ⟨suffix, hsuffix⟩
                    simp only [List.cons_append] at hsuffix
                    have hhead : trueLetter = falseLetter := by
                      injection hsuffix with hhead _htail
                      cases hhead
                      rfl
                    exact False.elim (hneq hhead)
                | loop_iter hrightBody hrightRest =>
                    rename_i rightBody rightRest
                    have hpTail : ListPrefix (leftBody ++ leftRest)
                        (rightBody ++ rightRest) :=
                      ListPrefix.cons_cancel hp
                    rcases hpTail with ⟨suffix, hsuffix⟩
                    have hlt : leftRest.length < m := by
                      rw [← hlen]
                      have hpos : 0 < leftBody.length + 1 :=
                        Nat.zero_lt_succ _
                      have hltBase :
                          leftRest.length <
                            (leftBody.length + 1) + leftRest.length :=
                        Nat.lt_add_of_pos_left (n := leftRest.length) hpos
                      simpa [List.length_append, Nat.add_assoc, Nat.add_comm,
                        Nat.add_left_comm] using hltBase
                    cases Nat.le_total leftBody.length rightBody.length with
                    | inl hLen =>
                        have hprefixBody : ListPrefix leftBody rightBody :=
                          ListPrefix.append_eq_left_prefix
                            (u1 := leftBody)
                            (u2 := leftRest ++ suffix)
                            (v1 := rightBody)
                            (v2 := rightRest)
                            (by simpa [List.append_assoc] using hsuffix)
                            hLen
                        have hEqBody : leftBody = rightBody :=
                          ihBody hguardBody hleftBody hrightBody hprefixBody
                        have hprefixRest : ListPrefix leftRest rightRest := by
                          refine ⟨suffix, ?_⟩
                          have hcancel :
                              leftBody ++ (leftRest ++ suffix) =
                                leftBody ++ rightRest := by
                            simpa [hEqBody, List.append_assoc] using hsuffix
                          exact List.append_cancel_left hcancel
                        have hEqRest : leftRest = rightRest :=
                          ih leftRest.length hlt leftRest rfl hleftRest
                            rightRest hrightRest hprefixRest
                        simp [hEqBody, hEqRest]
                    | inr hLen =>
                        have hprefixBody : ListPrefix rightBody leftBody :=
                          ListPrefix.append_eq_right_prefix
                            (u1 := leftBody)
                            (u2 := leftRest ++ suffix)
                            (v1 := rightBody)
                            (v2 := rightRest)
                            (by simpa [List.append_assoc] using hsuffix)
                            hLen
                        have hEqBody : rightBody = leftBody :=
                          ihBody hguardBody hrightBody hleftBody hprefixBody
                        have hprefixRest : ListPrefix leftRest rightRest := by
                          refine ⟨suffix, ?_⟩
                          have hcancel :
                              leftBody ++ (leftRest ++ suffix) =
                                leftBody ++ rightRest := by
                            simpa [hEqBody, List.append_assoc] using hsuffix
                          exact List.append_cancel_left hcancel
                        have hEqRest : leftRest = rightRest :=
                          ih leftRest.length hlt leftRest rfl hleftRest
                            rightRest hrightRest hprefixRest
                        simp [hEqBody, hEqRest]
      intro left right hleft hright hp
      exact hmain left.length left rfl hleft right hright hp
  | par branches ihBranches =>
      intro left right hleft hright hp
      cases hleft with
      | par hleftBranches =>
          rename_i leftTagged
          cases hright with
          | par hrightBranches =>
              rename_i rightTagged
              have hpTagged : ListPrefix leftTagged rightTagged :=
                listPrefix_map_reflects
                  (wrapTaggedDecorated_injective
                    (Lifeline := Lifeline) (Channel := Channel))
                  hp
              have hEqTagged : leftTagged = rightTagged := by
                apply listPrefix_eq_of_eraseBranch_eq hpTagged
                intro i
                have hpBranch :=
                  eraseBranch_preserves_prefix (i := i) hpTagged
                exact ihBranches i (hguard i)
                  (hleftBranches i) (hrightBranches i) hpBranch
              simp [hEqTagged]

theorem dtrace_localPrefixFree {Lifeline : Type u} {Channel : Type v}
    {program : Lifeline → LocProg Lifeline Channel}
    (hguard : ∀ A, GuardDistinct (program A)) :
    LocalPrefixFree (fun A word => DTrace (program A) word) := by
  intro A _ _ hleft hright hp
  exact dtrace_prefix_free (program A) (hguard A) hleft hright hp

theorem componentPrefixFree_dtrace {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    (hguard : ∀ i A, GuardDistinct (program i A)) :
    ComponentPrefixFreeOf
      (fun i A word => DTrace (program i A) word) := by
  intro i A _ _ hleft hright hp
  exact dtrace_prefix_free (program i A) (hguard i A) hleft hright hp

theorem dtrace_parloc_to_trace_parloc {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {branches :
      ParLocBranchesOf n Lifeline (DecoratedLetter Lifeline Channel)}
    {trace :
      ParLocTaggedTraceOf n Lifeline (DecoratedLetter Lifeline Channel)}
    (hpar : ParLocOf branches trace) :
    ParLoc
      (fun i A => (branches i A).map DecoratedLetter.toLocal)
      (fun A => (trace A).map (Tagged.mapVal DecoratedLetter.toLocal)) :=
  parlocOf_map DecoratedLetter.toLocal hpar

theorem dtrace_taggedParLocCompleteLocal_to_trace {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {A : Lifeline}
    {word : Word (Tagged (Fin n) (DecoratedLetter Lifeline Channel))}
    (hcomplete :
      TaggedParLocCompleteLocalOf
        (fun i A word => DTrace (program i A) word) A word) :
    TaggedParLocCompleteLocal
      (fun i A word => trace (program i A) word) A
      (word.map (Tagged.mapVal DecoratedLetter.toLocal)) := by
  rcases hcomplete with ⟨branches, herase, hbranchesComplete⟩
  refine ⟨fun i => (branches i).map DecoratedLetter.toLocal, ?_, ?_⟩
  · intro i
    calc
      eraseBranch i (word.map (Tagged.mapVal DecoratedLetter.toLocal))
          = (eraseBranch i word).map DecoratedLetter.toLocal :=
            eraseBranch_mapVal i DecoratedLetter.toLocal word
      _ = (branches i).map DecoratedLetter.toLocal := by rw [herase i]
  · intro i
    exact dtrace_to_trace (hbranchesComplete i)

theorem dtrace_taggedParLocComplete_to_trace {n : Nat}
    {Lifeline : Type u} {Channel : Type v}
    {program : Fin n → Lifeline → LocProg Lifeline Channel}
    {dtraceTuple :
      ParLocTaggedTraceOf n Lifeline (DecoratedLetter Lifeline Channel)}
    (hcomplete :
      TaggedParLocCompleteOf
        (fun i A word => DTrace (program i A) word) dtraceTuple) :
    TaggedParLocComplete
      (fun i A word => trace (program i A) word)
      (fun A => (dtraceTuple A).map
        (Tagged.mapVal DecoratedLetter.toLocal)) := by
  rcases hcomplete with ⟨branches, hpar, hbranchesComplete⟩
  refine
    ⟨fun i A => (branches i A).map DecoratedLetter.toLocal,
      dtrace_parloc_to_trace_parloc hpar, ?_⟩
  intro i A
  exact dtrace_to_trace (hbranchesComplete i A)

/--
The untagged local trace language with `par` is not prefix-free in general.

This counterexample is why the parallel Zipper proof keeps the branch-origin
tagging as a proof witness when it invokes prefix-freeness.
-/
private def counterA : LocalLetter Unit Unit :=
  LocalLetter.action

private def counterB : LocalLetter Unit Unit :=
  LocalLetter.tau

private def counterLeft : LocProg Unit Unit :=
  LocProg.branch counterB counterA (LocProg.letter counterA) LocProg.eps

private def counterRight : LocProg Unit Unit :=
  LocProg.letter counterB

private def counterBranches : Fin 2 → LocProg Unit Unit :=
  Fin.cases counterLeft (fun _ : Fin 1 => counterRight)

private def counterPar : LocProg Unit Unit :=
  LocProg.par counterBranches

private theorem counter_guardDistinct :
    GuardDistinct counterPar := by
  change ∀ i, GuardDistinct (counterBranches i)
  rw [Fin.forall_fin_two]
  constructor
  · simp [counterBranches, counterLeft, GuardDistinct]
    intro h
    cases h
  · exact trivial

private theorem counter_short :
    trace counterPar [counterB, counterA] := by
  let tagged : Word (Tagged (Fin 2) (LocalLetter Unit Unit)) :=
    [{ branch := (1 : Fin 2), val := counterB },
     { branch := (0 : Fin 2), val := counterA }]
  let branchWords : Fin 2 → Word (LocalLetter Unit Unit) :=
    Fin.cases [counterA] (fun _ : Fin 1 => [counterB])
  refine Trace.par (tagged := tagged) (branchWords := branchWords) ?_ ?_ ?_
  · rfl
  · rw [Fin.forall_fin_two]
    constructor <;> rfl
  · rw [Fin.forall_fin_two]
    constructor
    · exact Trace.branch_false Trace.eps
    · exact Trace.letter counterB

private theorem counter_long :
    trace counterPar [counterB, counterA, counterB] := by
  let tagged : Word (Tagged (Fin 2) (LocalLetter Unit Unit)) :=
    [{ branch := (0 : Fin 2), val := counterB },
     { branch := (0 : Fin 2), val := counterA },
     { branch := (1 : Fin 2), val := counterB }]
  let branchWords : Fin 2 → Word (LocalLetter Unit Unit) :=
    Fin.cases [counterB, counterA] (fun _ : Fin 1 => [counterB])
  refine Trace.par (tagged := tagged) (branchWords := branchWords) ?_ ?_ ?_
  · rfl
  · rw [Fin.forall_fin_two]
    constructor <;> rfl
  · rw [Fin.forall_fin_two]
    constructor
    · exact Trace.branch_true (Trace.letter counterA)
    · exact Trace.letter counterB

theorem trace_par_untagged_not_prefix_free :
    ∃ program : LocProg Unit Unit,
      GuardDistinct program ∧
      ¬ WordPrefixFree (fun word => trace program word) := by
  refine ⟨counterPar, counter_guardDistinct, ?_⟩
  intro hprefix
  have heq := hprefix counter_short counter_long ⟨[counterB], rfl⟩
  have hlen := congrArg List.length heq
  simp at hlen

end LocProg

/-- Distributed local program: one local program per lifeline. -/
abbrev DistProg (Lifeline : Type u) (Channel : Type v) :=
  Lifeline → LocProg Lifeline Channel

namespace DistProg

/-- Complete distributed local traces before imposing any MSC predicate. -/
def trace {Lifeline : Type u} {Channel : Type v}
    (program : DistProg Lifeline Channel)
    (words : Lifeline → Word (LocalLetter Lifeline Channel)) : Prop :=
  ∀ A, LocProg.trace (program A) (words A)

/-- Prefix-closed distributed local traces before imposing any MSC predicate. -/
def prefixTrace {Lifeline : Type u} {Channel : Type v}
    (program : DistProg Lifeline Channel)
    (words : Lifeline → Word (LocalLetter Lifeline Channel)) : Prop :=
  ∀ A, LocProg.prefixTrace (program A) (words A)

theorem trace_prefixTrace {Lifeline : Type u} {Channel : Type v}
    {program : DistProg Lifeline Channel}
    {words : Lifeline → Word (LocalLetter Lifeline Channel)}
    (h : trace program words) :
    prefixTrace program words := by
  intro A
  exact LocProg.trace_prefixTrace (h A)

/--
Distributed semantics parameterized by the MSC predicate.  The concrete MSC
construction is kept outside this local-syntax file.
-/
def semantics {Lifeline : Type u} {Channel : Type v}
    (IsCompleteMSC :
      (Lifeline → Word (LocalLetter Lifeline Channel)) → Prop)
    (program : DistProg Lifeline Channel)
    (words : Lifeline → Word (LocalLetter Lifeline Channel)) : Prop :=
  trace program words ∧ IsCompleteMSC words

/--
Distributed prefix semantics parameterized by the prefix-MSC predicate.  This is
the local side of the Zipper precondition.
-/
def prefixSemantics {Lifeline : Type u} {Channel : Type v}
    (IsMSC :
      (Lifeline → Word (LocalLetter Lifeline Channel)) → Prop)
    (program : DistProg Lifeline Channel)
    (words : Lifeline → Word (LocalLetter Lifeline Channel)) : Prop :=
  prefixTrace program words ∧ IsMSC words

end DistProg

end ZipperGenParallel
