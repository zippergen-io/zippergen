import ZipperGenSpawn.SpawnProvenance

namespace ZipperGenSpawn

universe u v

/-- Child names emitted at one spawn site on one participant, in local visit
    order. -/
def forkChildren (parent : SessionName) (site : Occurrence) :
    List (ProjectedSpawnLetter Lifeline ChannelScope) -> List SessionName
  | [] => []
  | .event _ _ :: rest => forkChildren parent site rest
  | .fork current currentSite child :: rest =>
      if current = parent /\ currentSite = site then
        child :: forkChildren parent site rest
      else
        forkChildren parent site rest

theorem forkCount_eq_length_forkChildren
    (parent : SessionName) (site : Occurrence)
    (history : List (ProjectedSpawnLetter Lifeline ChannelScope)) :
    forkCount parent site history = (forkChildren parent site history).length := by
  induction history with
  | nil => rfl
  | cons head rest ih =>
      cases head with
      | event session letter => simp [forkCount, forkChildren, ih]
      | fork current currentSite child =>
          simp only [forkCount, forkChildren, ih]
          split <;> simp <;> omega

theorem forkChildren_append
    (parent : SessionName) (site : Occurrence)
    (left right : List (ProjectedSpawnLetter Lifeline ChannelScope)) :
    forkChildren parent site (left ++ right) =
      forkChildren parent site left ++ forkChildren parent site right := by
  induction left with
  | nil => rfl
  | cons head rest ih =>
      cases head <;> simp [forkChildren, ih] <;> split <;> simp_all

/-- A participant's `k`th visit to `(parent,site)` has the canonical dynamic
    name `child(parent,site,k+1)`. -/
def CanonicalChildList (parent : SessionName) (site : Occurrence)
    (children : List SessionName) : Prop :=
  forall k child,
    children[k]? = some child ->
      child = SessionName.child parent site (k + 1)

theorem canonicalChildList_nil (parent : SessionName) (site : Occurrence) :
    CanonicalChildList parent site [] := by
  intro k child h
  simp at h

theorem canonicalChildList_append
    (parent : SessionName) (site : Occurrence)
    (children : List SessionName) (child : SessionName)
    (hchildren : CanonicalChildList parent site children)
    (hchild : child = SessionName.child parent site (children.length + 1)) :
    CanonicalChildList parent site (children ++ [child]) := by
  intro k current hlookup
  by_cases hk : k < children.length
  · have hold : children[k]? = some current := by
      rw [List.getElem?_append_left hk] at hlookup
      exact hlookup
    exact hchildren k current hold
  · have hkBound : k < children.length + 1 := by
      have := List.getElem?_eq_some_iff.mp hlookup
      simpa using this.1
    have hkEq : k = children.length := by omega
    subst k
    have hcurrent : current = child := by
      exact (by simpa using hlookup : child = current).symm
    subst current
    exact hchild

def ForkRanksCanonical
    (configuration : Configuration Lifeline Field) : Prop :=
  forall A parent site,
    CanonicalChildList parent site
      (forkChildren parent site (configuration.history A))

theorem initial_forkRanksCanonical [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field) :
    ForkRanksCanonical (initialConfiguration order source) := by
  intro A parent site
  exact canonicalChildList_nil parent site

theorem step_preserves_forkRanksCanonical [DecidableEq Lifeline]
    {before after : Configuration Lifeline Field}
    (hranks : ForkRanksCanonical before)
    (hcounters : CountersMatchHistory before)
    (hstep : Step before after) :
    ForkRanksCanonical after := by
  intro B parent site
  cases hstep with
  | run A session program residual counters' queues' emission activation
      hcontrol hlocal hfresh =>
      cases emission with
      | silent =>
          simpa [appendEmission] using hranks B parent site
      | event letter =>
          by_cases hBA : B = A
          · subst B
            simpa [appendEmission, forkChildren_append, forkChildren] using
              hranks A parent site
          · simpa [appendEmission, hBA] using hranks B parent site
      | fork emittedParent emittedSite emittedChild =>
          by_cases hBA : B = A
          · subst B
            have hparent : emittedParent = session :=
              localStep_fork_parent hlocal rfl
            have hchild : emittedChild = SessionName.child session emittedSite
                (nextVisit before.counters A session emittedSite) :=
              localStep_fork_child hlocal rfl
            subst emittedParent
            by_cases hp : parent = session
            · subst parent
              by_cases hs : site = emittedSite
              · subst site
                simp only [appendEmission, if_pos]
                rw [forkChildren_append]
                simp only [forkChildren]
                apply canonicalChildList_append
                · exact hranks A session emittedSite
                · rw [hchild, nextVisit, hcounters A session emittedSite,
                    forkCount_eq_length_forkChildren]
              · have hs' : emittedSite ≠ site := Ne.symm hs
                simpa [appendEmission, forkChildren_append, forkChildren,
                  hs, hs'] using hranks A session site
            · have hp' : session ≠ parent := Ne.symm hp
              simpa [appendEmission, forkChildren_append, forkChildren,
                hp, hp'] using hranks A parent site
          · simpa [appendEmission, hBA] using hranks B parent site

theorem reachable_forkRanksCanonical [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field) :
    forall configuration,
      Reachable (initialConfiguration order source) configuration ->
        ForkRanksCanonical configuration := by
  intro configuration hreachable
  induction hreachable with
  | refl => exact initial_forkRanksCanonical order source
  | @tail before after hbefore hstep ih =>
      exact step_preserves_forkRanksCanonical ih
        (reachable_countersMatchHistory order source before hbefore) hstep

/-- Corresponding ordinal visits agree on their child session name even when
    the physical participants reach those visits at different scheduler
    times. -/
theorem reachable_fork_children_agree [DecidableEq Lifeline]
    (order : List Lifeline) (source : Prog Lifeline Field)
    (configuration : Configuration Lifeline Field)
    (hreachable : Reachable (initialConfiguration order source) configuration)
    (A B : Lifeline) (parent : SessionName) (site : Occurrence)
    (k : Nat) (leftChild rightChild : SessionName)
    (hleft : (forkChildren parent site (configuration.history A))[k]? =
      some leftChild)
    (hright : (forkChildren parent site (configuration.history B))[k]? =
      some rightChild) :
    leftChild = rightChild := by
  have hranks := reachable_forkRanksCanonical order source configuration
    hreachable
  rw [hranks A parent site k leftChild hleft,
    hranks B parent site k rightChild hright]

end ZipperGenSpawn
