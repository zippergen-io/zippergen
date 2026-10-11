"""Single dependency policy shared by submission checks and supplied-helper gates."""
import json,re
import packet

# Spell out the round-decomposition rule: it is the alignment work under study.
EXPLICIT = [
    'SupplierAccess.global_round_completion',
    'realization_sound', 'realization_complete', 'projectDist_deadlockFree',
    'distSemantics_while_decompose', 'distSemantics_if_decompose', 'uniformZipper',
    'ZipperGenSpawn.projected_prefix_realization',
    'ZipperGenSpawn.projected_prefix_property',
    'ZipperGenSpawn.projected_msc_prefix_property',
    'ZipperGenSpawn.spawned_prefix_realization',
    'SupplierAccess.global_source_safety', 'SupplierAccess.global_safety',
    'SupplierAccess.global_projected_safety', 'SupplierAccess.global_certificate',
    'SupplierAccess.global_shapes', 'SupplierAccess.tagged_execution_lifting',
    'SupplierAccess.tagged_execution_lift_if',
    'SupplierAccess.universal_alignment_of_complete',
]
PREFIXES = ['uniformZipper_', 'realization_',
            '_private.MSCAgents.Correctness.', '_private.MSCAgents.DeadlockFreeness.']

def banned_names():
    names=set(EXPLICIT)
    for module in ('Correctness','DeadlockFreeness'):
        text=(packet.PIN/'MSCAgents'/f'{module}.lean').read_text()
        names.update(re.findall(r"(?m)^theorem\s+([A-Za-z_][A-Za-z_0-9']*)",text))
    return sorted(names)

def direct_for(arm, direct=None):
    return arm in ('local','tagged') if direct is None else direct

def lean_support():
    names=', '.join(json.dumps(n) for n in banned_names())
    prefixes=', '.join(json.dumps(n) for n in PREFIXES)
    return f'''
open Lean Elab Command
private def routeBanned (n : Name) : Bool :=
  ([{names}] : List String).contains n.toString ||
    ([{prefixes}] : List String).any (fun p => n.toString.startsWith p)

-- Types AND values are traversed. Supplied declarations and private wrappers
-- receive no exemption; an import alone is not a proof dependency.
private partial def pilotDependencies (env : Environment) (n : Name) (seen : NameSet) : NameSet := Id.run do
  if seen.contains n then return seen
  let mut seen := seen.insert n
  if let some info := env.find? n then
    for dep in info.getUsedConstantsAsSet do
      seen := pilotDependencies env dep seen
  return seen
'''

def submission_audit(direct):
    action=('throwError m!"Forbidden global proof dependency: {n}"' if direct else
            'logInfo m!"GLOBAL_ROUTE_DEPENDENCY: {n}"')
    return lean_support()+f'''
elab "#pilot_dependency_audit" : command => do
  let env ← getEnv
  let deps := pilotDependencies env ``checked_submission {{}}
  for n in deps do
    if routeBanned n then
      {action}
#pilot_dependency_audit
'''

def helper_audit(modules):
    names=', '.join(json.dumps(n) for n in sorted(modules))
    return lean_support()+f'''
elab "#supplied_helper_audit" : command => do
  let env ← getEnv
  let mut seen : NameSet := {{}}
  let mut roots := 0
  for (n, _) in env.constants.toList do
    if let some idx := env.getModuleIdxFor? n then
      let owner := env.header.moduleNames[idx.toNat]!
      if ([{names}] : List String).contains owner.toString then
        roots := roots + 1
        logInfo m!"HELPER_ROOT: {{owner}} {{n}}"
        seen := pilotDependencies env n seen
  if roots == 0 then throwError "No supplied declarations were audited"
  let mut failures := 0
  for n in seen do
    if routeBanned n then
      failures := failures + 1
      logError m!"SUPPLIED_HELPER_FORBIDDEN: {{n}}"
  if failures == 0 then logInfo m!"SUPPLIED_HELPERS_CLEAN: {{roots}}"
#supplied_helper_audit
'''
