import CompletionGoal

set_option maxRecDepth 8192
set_option maxHeartbeats 8000000

namespace ReviewCompletion
open SupplierAccess

private def noLoops {a : Role} : Local a → Prop
  | .eps | .send .. | .recv .. | .act .. => True
  | .seq p q => noLoops p ∧ noLoops q
  | .recvIf _ _ p q | .localIf _ p q => noLoops p ∧ noLoops q
  | .recvWhile .. | .localWhile .. => False

private def traces {a : Role} : Local a → List (Word a)
  | .eps => [[]]
  | .send x b => if h : a ≠ b then [[AlphabetOf.mkSend a x b h]] else []
  | .recv x b => [[AlphabetOf.mkRecv a x b]]
  | .act y f x => [[AlphabetOf.mkAct a y f x]]
  | .seq p q => (traces p).flatMap (fun u => (traces q).map (u ++ ·))
  | .recvIf x b p q =>
      ((traces p).map (AlphabetOf.mkRecv a (ControlPayload.setDecision true x) b :: ·)) ++
      ((traces q).map (AlphabetOf.mkRecv a (ControlPayload.setDecision false x) b :: ·))
  | .localIf c p q =>
      ((traces p).map (AlphabetOf.mkIfTrue a c :: ·)) ++
      ((traces q).map (AlphabetOf.mkIfFalse a c :: ·))
  | .recvWhile .. | .localWhile .. => []

private theorem traces_sound {a : Role} (p : Local a) (w : Word a)
    (h : w ∈ traces p) : localTraceSemantics p w := by
  induction p generalizing w with
  | eps => simpa [traces, localTraceSemantics] using h
  | send x b =>
      by_cases hab : a ≠ b
      · simpa [traces, hab, localTraceSemantics] using h
      · simp [traces, hab] at h
  | recv x b => simpa [traces, localTraceSemantics] using h
  | act y f x => simpa [traces, localTraceSemantics] using h
  | seq p q ihp ihq =>
      simp only [traces, List.mem_flatMap, List.mem_map] at h
      rcases h with ⟨u, hu, v, hv, rfl⟩
      exact ⟨u, v, ihp _ hu, ihq _ hv, rfl⟩
  | recvIf x b p q ihp ihq =>
      simp only [traces, List.mem_append, List.mem_map] at h
      rcases h with ⟨u, hu, rfl⟩ | ⟨u, hu, rfl⟩
      · exact Or.inl ⟨u, ihp _ hu, rfl⟩
      · exact Or.inr ⟨u, ihq _ hu, rfl⟩
  | localIf c p q ihp ihq =>
      simp only [traces, List.mem_append, List.mem_map] at h
      rcases h with ⟨u, hu, rfl⟩ | ⟨u, hu, rfl⟩
      · exact Or.inl ⟨u, ihp _ hu, rfl⟩
      · exact Or.inr ⟨u, ihq _ hu, rfl⟩
  | recvWhile => simp [traces] at h
  | localWhile => simp [traces] at h

private theorem traces_complete {a : Role} (p : Local a) (w : Word a)
    (hn : noLoops p) (h : localTraceSemantics p w) : w ∈ traces p := by
  induction p generalizing w with
  | eps => simpa [traces, localTraceSemantics] using h
  | send x b =>
      rcases h with ⟨hab, rfl⟩
      simp [traces, hab]
  | recv x b => simpa [traces, localTraceSemantics] using h
  | act y f x => simpa [traces, localTraceSemantics] using h
  | seq p q ihp ihq =>
      rcases hn with ⟨hnp, hnq⟩
      rcases h with ⟨u, v, hu, hv, rfl⟩
      simp only [traces, List.mem_flatMap, List.mem_map]
      exact ⟨u, ihp _ hnp hu, v, ihq _ hnq hv, rfl⟩
  | recvIf x b p q ihp ihq =>
      rcases hn with ⟨hnp, hnq⟩
      rcases h with ⟨u, hu, rfl⟩ | ⟨u, hu, rfl⟩
      · simp [traces, ihp _ hnp hu]
      · simp [traces, ihq _ hnq hu]
  | localIf c p q ihp ihq =>
      rcases hn with ⟨hnp, hnq⟩
      rcases h with ⟨u, hu, rfl⟩ | ⟨u, hu, rfl⟩
      · simp [traces, ihp _ hnp hu]
      · simp [traces, ihq _ hnq hu]
  | recvWhile => simp [noLoops] at hn
  | localWhile => simp [noLoops] at hn

private theorem endpoints_noLoops (a : Role) : noLoops (endpoints a) := by
  cases a <;> simp [endpoints, noLoops]

private def run {a : Role} : Local a → List Port → Option (Word a × List Port)
  | .eps, xs => some ([], xs)
  | .send x b, xs =>
      if h : a ≠ b then some ([AlphabetOf.mkSend a x b h], xs) else none
  | .recv x b, xs =>
      if b = .agent then
        match xs with
        | y :: ys => if x = y then some ([AlphabetOf.mkRecv a x b], ys) else none
        | [] => none
      else some ([AlphabetOf.mkRecv a x b], xs)
  | .act y f x, xs => some ([AlphabetOf.mkAct a y f x], xs)
  | .seq p q, xs => do
      let (u, ys) ← run p xs
      let (v, zs) ← run q ys
      pure (u ++ v, zs)
  | .recvIf x b p q, xs =>
      if b = .agent then
        match xs with
        | y :: ys =>
            if y = ControlPayload.setDecision true x then do
              let (u, zs) ← run p ys
              pure (AlphabetOf.mkRecv a y b :: u, zs)
            else if y = ControlPayload.setDecision false x then do
              let (u, zs) ← run q ys
              pure (AlphabetOf.mkRecv a y b :: u, zs)
            else none
        | [] => none
      else none
  | .localIf .., _ => none
  | .recvWhile .., _ => none
  | .localWhile .., _ => none

private def candidate (ag : Word .agent) (a : Role) : Word a :=
  if h : a = .agent then h ▸ ag
  else ((run (endpoints a) (sendPayloads a ag)).map Prod.fst).getD []

private theorem run_sound {a : Role} (p : Local a) :
    ∀ xs w ys, run p xs = some (w, ys) → localTraceSemantics p w := by
  induction p with
  | eps =>
      intro xs w ys h
      simp [run] at h
      rcases h with ⟨rfl, rfl⟩
      rfl
  | send x b =>
      intro xs w ys h
      by_cases hab : a ≠ b
      · simp [run, hab] at h
        rcases h with ⟨rfl, rfl⟩
        exact ⟨hab, rfl⟩
      · simp [run, hab] at h
  | recv x b =>
      intro xs w ys h
      by_cases hb : b = .agent
      · subst b
        cases xs with
        | nil => simp [run] at h
        | cons z zs =>
            by_cases hx : x = z
            · simp [run, hx] at h
              rcases h with ⟨rfl, rfl⟩
              cases hx
              rfl
            · simp [run, hx] at h
      · simp [run, hb] at h
        rcases h with ⟨rfl, rfl⟩
        rfl
  | act y f x =>
      intro xs w ys h
      simp [run] at h
      rcases h with ⟨rfl, rfl⟩
      rfl
  | seq p q ihp ihq =>
      intro xs w ys h
      cases hp : run p xs with
      | none => simp [run, hp] at h
      | some pw =>
          obtain ⟨u, zs⟩ := pw
          cases hq : run q zs with
          | none => simp [run, hp, hq] at h
          | some qw =>
              obtain ⟨v, ts⟩ := qw
              simp [run, hp, hq] at h
              rcases h with ⟨rfl, rfl⟩
              exact ⟨u, v, ihp xs u zs hp, ihq zs v ts hq, rfl⟩
  | recvIf x b p q ihp ihq =>
      intro xs w ys h
      by_cases hb : b = .agent
      · subst b
        cases xs with
        | nil => simp [run] at h
        | cons z zs =>
            by_cases ht : z = ControlPayload.setDecision true x
            · cases hp : run p zs with
              | none => simp [run, ht, hp] at h
              | some pw =>
                  obtain ⟨u, ts⟩ := pw
                  simp [run, ht, hp] at h
                  rcases h with ⟨rfl, rfl⟩
                  exact Or.inl ⟨u, ihp zs u ts hp, by simp [ht]⟩
            · by_cases hf : z = ControlPayload.setDecision false x
              · cases hq : run q zs with
                | none => simp [run, ht, hf, hq, ControlPayload.setDecision] at h
                | some qw =>
                    obtain ⟨u, ts⟩ := qw
                    simp [run, ht, hf, hq, ControlPayload.setDecision] at h
                    rcases h with ⟨rfl, rfl⟩
                    exact Or.inr ⟨u, ihq zs u ts hq, by simp [ControlPayload.setDecision]⟩
              · simp [run, ht, hf] at h
      · simp [run, hb] at h
  | localIf => intro xs w ys h; simp [run] at h
  | recvWhile => intro xs w ys h; simp [run] at h
  | localWhile => intro xs w ys h; simp [run] at h

private theorem run_stream {a : Role} (p : Local a) :
    ∀ xs w ys, run p xs = some (w, ys) →
      xs = recvPayloads .agent w ++ ys := by
  induction p with
  | eps =>
      intro xs w ys h
      simp [run] at h
      rcases h with ⟨rfl, rfl⟩
      rfl
  | send x b =>
      intro xs w ys h
      by_cases hab : a ≠ b
      · simp [run, hab] at h
        rcases h with ⟨rfl, rfl⟩
        simp [recvPayloads, AlphabetOf.mkSend]
      · simp [run, hab] at h
  | recv x b =>
      intro xs w ys h
      by_cases hb : b = .agent
      · subst b
        cases xs with
        | nil => simp [run] at h
        | cons z zs =>
            by_cases hx : x = z
            · simp [run, hx] at h
              rcases h with ⟨rfl, rfl⟩
              simp [recvPayloads, AlphabetOf.mkRecv, hx]
            · simp [run, hx] at h
      · simp [run, hb] at h
        rcases h with ⟨rfl, rfl⟩
        simp [recvPayloads, AlphabetOf.mkRecv, hb]
  | act y f x =>
      intro xs w ys h
      simp [run] at h
      rcases h with ⟨rfl, rfl⟩
      simp [recvPayloads, AlphabetOf.mkAct]
  | seq p q ihp ihq =>
      intro xs w ys h
      cases hp : run p xs with
      | none => simp [run, hp] at h
      | some pw =>
          obtain ⟨u, zs⟩ := pw
          cases hq : run q zs with
          | none => simp [run, hp, hq] at h
          | some qw =>
              obtain ⟨v, ts⟩ := qw
              simp [run, hp, hq] at h
              rcases h with ⟨rfl, rfl⟩
              simp [ihp xs u zs hp, ihq zs v ts hq, List.append_assoc]
  | recvIf x b p q ihp ihq =>
      intro xs w ys h
      by_cases hb : b = .agent
      · subst b
        cases xs with
        | nil => simp [run] at h
        | cons z zs =>
            by_cases ht : z = ControlPayload.setDecision true x
            · cases hp : run p zs with
              | none => simp [run, ht, hp] at h
              | some pw =>
                  obtain ⟨u, ts⟩ := pw
                  simp [run, ht, hp] at h
                  rcases h with ⟨rfl, rfl⟩
                  simp [recvPayloads, AlphabetOf.mkRecv, ht, ihp zs u ts hp]
            · by_cases hf : z = ControlPayload.setDecision false x
              · cases hq : run q zs with
                | none => simp [run, ht, hf, hq, ControlPayload.setDecision] at h
                | some qw =>
                    obtain ⟨u, ts⟩ := qw
                    simp [run, ht, hf, hq, ControlPayload.setDecision] at h
                    rcases h with ⟨rfl, rfl⟩
                    simp [recvPayloads, AlphabetOf.mkRecv, ControlPayload.setDecision,
                      ihq zs u ts hq, hf]
              · simp [run, ht, hf] at h
      · simp [run, hb] at h
  | localIf => intro xs w ys h; simp [run] at h
  | recvWhile => intro xs w ys h; simp [run] at h
  | localWhile => intro xs w ys h; simp [run] at h

private def controlled {a : Role} : Local a → Prop
  | .eps | .send .. | .recv .. | .act .. => True
  | .seq p q => controlled p ∧ controlled q
  | .recvIf _ b p q => b = .agent ∧ controlled p ∧ controlled q
  | .localIf .. | .recvWhile .. | .localWhile .. => False

private theorem controlled_endpoints (a : Role) (ha : a ≠ .agent) :
    controlled (endpoints a) := by
  cases a <;> simp_all [endpoints, controlled]

private def Pref {α : Type} (u v : List α) : Prop := ∃ t, v = u ++ t

private theorem prefix_trans {α : Type} {u v w : List α}
    (h₁ : Pref u v) (h₂ : Pref v w) : Pref u w := by
  rcases h₁ with ⟨x, rfl⟩
  rcases h₂ with ⟨y, rfl⟩
  exact ⟨x ++ y, by simp [List.append_assoc]⟩

private theorem prefix_append_left {α : Type} (u v : List α) :
    Pref u (u ++ v) := ⟨v, rfl⟩

private theorem prefix_append_right {α : Type} {u v : List α}
    (h : Pref u v) (t : List α) : Pref u (v ++ t) :=
  prefix_trans h (prefix_append_left v t)

private theorem prefix_cons_cancel {α : Type} {x : α} {u v : List α}
    (h : Pref (x :: u) (x :: v)) : Pref u v := by
  rcases h with ⟨t, ht⟩
  exact ⟨t, by simpa using congrArg List.tail ht⟩

private theorem run_prefix {a : Role} (p : Local a) (hc : controlled p) :
    ∀ xs w ys u, run p xs = some (w, ys) →
      localPrefixSemantics p u →
      Pref (recvPayloads .agent u) xs →
      Pref u w := by
  induction p with
  | eps =>
      intro xs w ys u hr hu hs
      simp [run] at hr
      rcases hr with ⟨rfl, rfl⟩
      rcases hu with ⟨v, hv, hp⟩
      simp [localTraceSemantics] at hv
      subst v
      rcases hp with ⟨t, ht⟩
      have he : u = [] := by
        cases u with
        | nil => rfl
        | cons hd tl => simp at ht
      subst u
      exact ⟨[], rfl⟩
  | send x b =>
      intro xs w ys u hr hu hs
      by_cases hab : a ≠ b
      · simp [run, hab] at hr
        rcases hr with ⟨rfl, rfl⟩
        rcases hu with ⟨v, hv, hp⟩
        rcases hv with ⟨_, rfl⟩
        exact hp
      · simp [run, hab] at hr
  | recv x b =>
      intro xs w ys u hr hu hs
      rcases hu with ⟨v, hv, hp⟩
      simp [localTraceSemantics] at hv
      subst v
      by_cases hb : b = .agent
      · subst b
        cases xs with
        | nil => simp [run] at hr
        | cons z zs =>
            by_cases hx : x = z
            · simp [run, hx] at hr
              rcases hr with ⟨rfl, rfl⟩
              cases hx
              exact hp
            · simp [run, hx] at hr
      · simp [run, hb] at hr
        rcases hr with ⟨rfl, rfl⟩
        exact hp
  | act y f x =>
      intro xs w ys u hr hu hs
      simp [run] at hr
      rcases hr with ⟨rfl, rfl⟩
      rcases hu with ⟨v, hv, hp⟩
      simp [localTraceSemantics] at hv
      subst v
      exact hp
  | seq p q ihp ihq =>
      rcases hc with ⟨hcp, hcq⟩
      intro xs w ys u hr hu hs
      cases hp : run p xs with
      | none => simp [run, hp] at hr
      | some pw =>
          obtain ⟨w₁, xs₁⟩ := pw
          cases hq : run q xs₁ with
          | none => simp [run, hp, hq] at hr
          | some qw =>
              obtain ⟨w₂, xs₂⟩ := qw
              simp [run, hp, hq] at hr
              rcases hr with ⟨rfl, rfl⟩
              rcases localPrefixSemantics_seq_split p q u hu with
                ⟨u₁, u₂, rfl, hu₁, hu₂, hfull⟩
              have hs₁ : Pref (recvPayloads .agent u₁) xs := by
                apply prefix_trans
                  (prefix_append_left (recvPayloads .agent u₁)
                    (recvPayloads .agent u₂))
                simpa [recvPayloads_append] using hs
              have hpref₁ := ihp hcp xs w₁ xs₁ u₁ hp hu₁ hs₁
              by_cases he : u₂ = []
              · subst u₂
                simpa using prefix_append_right hpref₁ w₂
              · have htrace₁ := hfull he
                have heq : u₁ = w₁ :=
                  localTraceSemantics_prefix_eq p htrace₁
                    (run_sound p xs w₁ xs₁ hp) hpref₁
                subst u₁
                rcases hs with ⟨t, ht⟩
                have hxs := run_stream p xs w₁ xs₁ hp
                have hxs₁ : xs₁ = recvPayloads .agent u₂ ++ t := by
                  have hh : recvPayloads .agent w₁ ++ xs₁ =
                      recvPayloads .agent w₁ ++ (recvPayloads .agent u₂ ++ t) := by
                    simpa [recvPayloads_append, List.append_assoc] using hxs.symm.trans ht
                  exact List.append_cancel_left hh
                have hpref₂ := ihq hcq xs₁ w₂ xs₂ u₂ hq hu₂ ⟨t, hxs₁⟩
                rcases hpref₂ with ⟨t₂, ht₂⟩
                exact ⟨t₂, by simpa [List.append_assoc] using congrArg (w₁ ++ ·) ht₂⟩
  | recvIf x b p q ihp ihq =>
      rcases hc with ⟨hb, hcp, hcq⟩
      subst b
      intro xs w ys u hr hu hs
      rcases localPrefixSemantics_recvIf_cases x .agent p q u hu with
        he | ⟨u', hu', he⟩ | ⟨u', hu', he⟩
      · subst u
        exact ⟨w, by simp⟩
      · subst u
        have hs' : Pref (ControlPayload.setDecision true x ::
            recvPayloads .agent u') xs := by
          simpa [recvPayloads, AlphabetOf.mkRecv] using hs
        rcases hs' with ⟨tail, htail⟩
        cases xs with
        | nil => simp at htail
        | cons z zs =>
            have hz : z = ControlPayload.setDecision true x :=
              (List.cons.inj htail).1
            have hzs : zs = recvPayloads .agent u' ++ tail :=
              (List.cons.inj htail).2
            subst z
            cases hp : run p zs with
            | none => simp [run, hp, ControlPayload.setDecision] at hr
            | some pw =>
                obtain ⟨v, ts⟩ := pw
                simp [run, hp, ControlPayload.setDecision] at hr
                rcases hr with ⟨rfl, rfl⟩
                have hv := ihp hcp zs v ts u' hp hu' ⟨tail, hzs⟩
                rcases hv with ⟨t, ht⟩
                exact ⟨t, by simp [ht, ControlPayload.setDecision]⟩
      · subst u
        have hs' : Pref (ControlPayload.setDecision false x ::
            recvPayloads .agent u') xs := by
          simpa [recvPayloads, AlphabetOf.mkRecv] using hs
        rcases hs' with ⟨tail, htail⟩
        cases xs with
        | nil => simp at htail
        | cons z zs =>
            have hz : z = ControlPayload.setDecision false x :=
              (List.cons.inj htail).1
            have hzs : zs = recvPayloads .agent u' ++ tail :=
              (List.cons.inj htail).2
            subst z
            cases hq : run q zs with
            | none => simp [run, hq, ControlPayload.setDecision] at hr
            | some qw =>
                obtain ⟨v, ts⟩ := qw
                simp [run, hq, ControlPayload.setDecision] at hr
                rcases hr with ⟨rfl, rfl⟩
                have hv := ihq hcq zs v ts u' hq hu' ⟨tail, hzs⟩
                rcases hv with ⟨t, ht⟩
                exact ⟨t, by simp [ht, ControlPayload.setDecision]⟩
  | localIf => simp [controlled] at hc
  | recvWhile => simp [controlled] at hc
  | localWhile => simp [controlled] at hc

private def success (ag : Word .agent) (a : Role) : Bool :=
  if a = .agent then true
  else match run (endpoints a) (sendPayloads a ag) with
       | some (_, []) => true
       | _ => false

private theorem success_all :
    ∀ ag ∈ traces (endpoints .agent), ∀ a : Role,
      success ag a = true := by
  decide

private theorem candidate_run {ag : Word .agent}
    (hag : ag ∈ traces (endpoints .agent)) {a : Role} (ha : a ≠ .agent) :
    run (endpoints a) (sendPayloads a ag) = some (candidate ag a, []) := by
  have hs := success_all ag hag a
  simp [success, ha] at hs
  cases hr : run (endpoints a) (sendPayloads a ag) with
  | none => simp [hr] at hs
  | some p =>
      obtain ⟨w, rest⟩ := p
      cases rest with
      | nil => simpa [candidate, ha, hr] using hr
      | cons x xs => simp [hr] at hs

private theorem channel_eq_all :
    ∀ ag ∈ traces (endpoints .agent), ∀ a b : Role,
      sendPayloads b (candidate ag a) =
      recvPayloads a (candidate ag b) := by
  decide

private def stage : Role → Nat
  | .agent => 0
  | .finalizer => 2
  | _ => 1

private theorem trace_shape_all :
    ∀ a : Role, ∀ w ∈ traces (endpoints a),
      w.length < 100 ∧
      ∀ b : Role, ∀ l ∈ w,
        l.val.isSendTo b = true → stage a < stage b := by
  decide

private theorem zip_prefix {α : Type} (xs ys : List α)
    (hlen : ys.length ≤ xs.length)
    (heq : ∀ p ∈ List.zip xs ys, p.1 = p.2) : Pref ys xs := by
  induction ys generalizing xs with
  | nil => exact ⟨xs, by simp⟩
  | cons y ys ih =>
      cases xs with
      | nil => simp at hlen
      | cons x xs =>
          have hxy : x = y := heq (x, y) (by simp)
          have htail : ∀ p ∈ List.zip xs ys, p.1 = p.2 := by
            intro p hp
            exact heq p (by simp [hp])
          rcases ih xs (by simpa using hlen) htail with ⟨t, ht⟩
          subst x
          exact ⟨t, by simp [ht]⟩

private theorem queue_prefix (M : Tuple) (hM : IsMSC M) (a b : Role) :
    Pref (recvPayloads a (M b)) (sendPayloads b (M a)) := by
  apply zip_prefix
  · simpa [sendPayloads_length, recvPayloads_length, sndCount, rcvCount] using
      hM.noUnmatchedRecv a b
  · intro p hp
    exact hM.labelCompat a b p hp

private theorem candidate_mem {ag : Word .agent}
    (hag : ag ∈ traces (endpoints .agent)) (a : Role) :
    candidate ag a ∈ traces (endpoints a) := by
  by_cases ha : a = .agent
  · subst a
    simpa [candidate] using hag
  · exact traces_complete (endpoints a) (candidate ag a)
      (endpoints_noLoops a)
      (run_sound (endpoints a) _ _ _ (candidate_run hag ha))

private theorem candidate_complete {ag : Word .agent}
    (hag : ag ∈ traces (endpoints .agent)) :
    IsCompleteMSC (candidate ag) := by
  refine ⟨?_, ?_, ?_⟩
  · intro a b
    have heq := channel_eq_all ag hag a b
    have hlen := congrArg List.length heq
    simpa [channelComplete, sndCount, rcvCount] using hlen
  · intro a b p hp
    have heq := channel_eq_all ag hag a b
    have aux : ∀ xs : List Port, ∀ q ∈ List.zip xs xs, q.1 = q.2 := by
      intro xs
      induction xs with
      | nil => intro q hq; simp at hq
      | cons x xs ih =>
          intro q hq
          simp at hq
          rcases hq with hq | hq
          · cases hq; rfl
          · exact ih q hq
    change p.1 = p.2
    exact aux _ p (by simpa [channelLabelCompatible, heq] using hp)
  · refine ⟨{
      rank := fun e => 100 * stage e.lifeline + e.pos
      local_mono := ?_
      fifo_mono := ?_
      }⟩
    · intro e₁ e₂ he hp _ _
      simp only [he]
      omega
    · intro a b j₁ j₂ hj₁ hj₂ hs hr hc
      have hshape := trace_shape_all a (candidate ag a) (candidate_mem hag a)
      have hstage : stage a < stage b :=
        hshape.2 b ((candidate ag a).get ⟨j₁, hj₁⟩)
          (List.get_mem _ _) hs
      have hbound : (candidate ag a).length < 100 := hshape.1
      dsimp
      omega

end ReviewCompletion

open ReviewCompletion SupplierAccess

theorem submission : ReviewCompletion.CompletionGoal := by
  intro M hM
  rcases hM with ⟨hloc, hmsc⟩
  rcases hloc .agent with ⟨ag, htrace, hprefix⟩
  have hag : ag ∈ traces (endpoints .agent) :=
    traces_complete (endpoints .agent) ag (endpoints_noLoops .agent) htrace
  refine ⟨candidate ag, ?_, ?_⟩
  · constructor
    · intro a
      by_cases ha : a = .agent
      · subst a
        simpa [candidate] using htrace
      · exact run_sound (endpoints a) _ _ _ (candidate_run hag ha)
    · exact candidate_complete hag
  · intro a
    by_cases ha : a = .agent
    · subst a
      simpa [candidate] using hprefix
    · have hrecv := queue_prefix M hmsc .agent a
      rcases hprefix with ⟨t, ht⟩
      have hsend : Pref (sendPayloads a (M .agent))
          (sendPayloads a ag) := by
        refine ⟨sendPayloads a t, ?_⟩
        simp [ht]
      have hstream := prefix_trans hrecv hsend
      have hp := run_prefix (endpoints a) (controlled_endpoints a ha)
        (sendPayloads a ag) (candidate ag a) [] (M a)
        (candidate_run hag ha) (hloc a) hstream
      exact hp
