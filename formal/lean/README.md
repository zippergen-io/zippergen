# Spawn formalization

Lean 4 development accompanying the ZipperGen spawn paper.

This is the canonical Lean development for ZipperGen's reusable proof
machinery. It contains the checked MSC and parallel foundations together with
the evolving spawn syntax, projection, session semantics, and safety-transfer
results.
The `zippergen-lean` repository retains the frozen developments accompanying
individual papers.

Current checked modules:

- `ZipperGenParallel.*` is the checked parallel-projection, shuffle, Zipper,
  and prefix-completion development imported from the parallel paper. The
  copy currently corresponds to paper commit
  `a8a625fe871d58a42bfe827ac96f6644d5ac40e0`.
- `ZipperGenSpawn.PrefixSafety` proves that every finite projected endpoint
  execution erases to a prefix of a global MSC and transfers prefix-closed
  global properties to unfinished distributed executions.
- `ZipperGenSpawn.Execution` defines session-tagged physical lifelines with
  local fork markers and lifts prefix realization and safety pointwise to every
  active spawned session.
- `ZipperGenSpawn.SourceSyntax`, `LocalProgram`, and `Projection` define the
  extended choreography, local fork templates, flat branch scopes, structural
  projection, and selective freshness metadata.
- `ZipperGenSpawn.SessionAgreement` and `OperationalNames` define canonical
  names `child(parent, site, visit)` and prove their injectivity and the naming
  consequence of ordinal-visit agreement.
- `ZipperGenSpawn.FlatScopes` and `CoreCompatibility` prove that the new flat
  branch scopes preserve the older path-labelled projection on every
  spawn-free program.
- `ZipperGenSpawn.ParallelPrefixSafety` and `SpawnPrefixSafety` transfer
  prefix-closed properties to unfinished executions and lift completion
  pointwise across all active sessions.
- `ZipperGenSpawn.OperationalSemantics` defines an executable-shape symbolic
  endpoint machine with session-indexed controls, participant-local spawn
  counters, session/scope-indexed FIFO queues, local parallel scheduling, and
  atomic child activation. It proves for every reachable configuration that
  child names are canonical and every child event follows that participant's
  own local fork; no global fork barrier is assumed.
- `ZipperGenSpawn.OperationalLocalTrace` proves that every symbolic local step,
  including a step inside a local parallel region, is a trace decomposition
  after detached child templates are erased. Consequently every finite local
  execution is a prefix of a complete trace of its initial endpoint program.
- `ZipperGenSpawn.OperationalControlTrace` carries those local witnesses through
  the network machine. Every allocated participant/session control is paired
  with exactly its session-restricted physical history; child creation starts
  from an empty child history, and every reachable root restriction is a prefix
  of the participant's actual projected source endpoint.
- `ZipperGenSpawn.OperationalCorrespondence` proves that runtime visit counters
  equal the number of earlier local forks, isolates equal prior counts as the
  exact child-name agreement obligation, and maintains the session/scope FIFO
  equation `sent = received ++ queue`. Thus every receive history is a prefix
  of the matching send history. These facts and the control/trace results are
  collected in one reachable-state invariant.
- `ZipperGenSpawn.TimedMSC` isolates the generic acyclicity argument.  A word
  tuple equipped with FIFO message matching and scheduler timestamps that
  increase along local and message edges is an MSC; acyclicity follows from
  strict increase around every causal path.
- `ZipperGenSpawn.OperationalSchedule` extracts a single ghost scheduler trace
  from every reachable endpoint configuration.  Restricting that trace by
  participant and session is exactly the corresponding physical session word,
  and its positions give timestamps that strictly increase along every
  participant-local successor edge.
- `ZipperGenSpawn.FIFORank` proves the generic rank and lookup facts used to
  pair selected sends and receives in participant-local words.
- `ZipperGenSpawn.OperationalMSC` constructs the FIFO message matching of
  every reachable session restriction.  Scheduler positions strictly increase
  along both local and message edges, so every restriction is an acyclic word
  MSC.  Together with the local trace theorem, this proves full prefix
  semantics for every unfinished root execution.
- `ZipperGenSpawn.SpawnProvenance` traces every projected fork template and
  allocated child control back to its statically declared spawn body, and
  preserves that origin through endpoint reduction.
- `ZipperGenSpawn.ForkAgreement` proves that the `k`th local visit to a spawn
  site has the canonical name `child(parent,site,k+1)`. Corresponding ordinal
  visits therefore agree even when participants reach them at different
  scheduler times.
- `ZipperGenSpawn.OperationalSpawnPrefix` combines provenance, local prefix
  traces, FIFO matching, acyclicity, and fork causality.  For source programs
  with globally unique control labels, every reachable physical execution
  satisfies the full `ProjectedSpawnPrefixSemantics` for all active root and
  child sessions.  It also exposes the resulting prefix-closed property
  transfer as `reachable_active_sessions_prefix_property`.

No module in `ZipperGenSpawn` or `ZipperGenParallel` uses `sorry` or declares
an axiom.  The endpoint machine now has a complete execution-side spawn
correspondence theorem: every reachable physical prefix has local fork
causality, every active session is a prefix of the projection of its source
body, and every session restriction is a valid FIFO MSC.  The structural lemma
`wellLabelled_spawnSpecsFunctional` proves that the existing global occurrence
uniqueness check makes each spawn site select exactly one freshness declaration
and source body, so the final theorem assumes only `Prog.WellLabelled`.

Build with:

```sh
lake build
```
