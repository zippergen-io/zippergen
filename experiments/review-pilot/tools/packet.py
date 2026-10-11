"""Matched one-request optional-review completion packets."""
from pathlib import Path
import hashlib,json,shutil
HERE=Path(__file__).resolve().parents[1]
PIN=HERE/'pinned-lib'
SIZES=(1,4,8)
EXTENSION_SIZES=(16,)
ALL_SIZES=SIZES+EXTENSION_SIZES
ARMS=('global','local')
COMMON=('SupplierVocabulary.lean','SupplierEndpoints.lean','CompletionGoal.lean')
GLOBAL_ONLY=('SupplierProtocol.lean','SupplierProjection.lean')
GOAL='ReviewCompletion.CompletionGoal'
def given_files(arm):
    if arm not in ARMS: raise ValueError(arm)
    return COMMON+(GLOBAL_ONLY if arm=='global' else ())
def verify_frozen():
    root=HERE/'frozen'
    if not json.loads((root/'READY.json').read_text()).get('ready'): raise RuntimeError('not frozen')
    for p,h in json.loads((root/'hashes.json').read_text()).items():
        if hashlib.sha256((root/p).read_bytes()).hexdigest()!=h: raise RuntimeError('frozen file changed: '+p)
    for p,h in json.loads((root/'library-manifest.json').read_text()).items():
        if hashlib.sha256((PIN/p).read_bytes()).hexdigest()!=h: raise RuntimeError('pinned library changed: '+p)
def source_root(k):
    if k in SIZES: return HERE/'frozen'
    if k in EXTENSION_SIZES: return HERE/f'frozen-k{k}'
    raise ValueError(k)
def verify_size(k):
    verify_frozen()
    root=source_root(k)
    if k in SIZES: return
    if not json.loads((root/'READY.json').read_text()).get('ready'): raise RuntimeError('extension not frozen')
    parent=hashlib.sha256((HERE/'frozen/hashes.json').read_bytes()).hexdigest()
    if (root/'BASE-HASHES.sha256').read_text().strip()!=parent: raise RuntimeError('extension base changed')
    for p,h in json.loads((root/'hashes.json').read_text()).items():
        if hashlib.sha256((root/p).read_bytes()).hexdigest()!=h: raise RuntimeError('extension file changed: '+p)
def build(k,arm,destination,*,frozen=True,copy_library=False):
    if k not in ALL_SIZES: raise ValueError(k)
    verify_size(k)
    source=source_root(k)
    destination.mkdir(parents=True,exist_ok=False)
    for name in given_files(arm): shutil.copy2(source/f'k{k}'/name,destination/name)
    shutil.copy2(HERE/'frozen'/'lean-toolchain',destination/'lean-toolchain')
    if arm=='local': shutil.copy2(HERE/'frozen'/'BANNED-THEOREMS.txt',destination/'BANNED-THEOREMS.txt')
    roots=', '.join('`'+Path(n).stem for n in given_files(arm))
    (destination/'lakefile.lean').write_text('import Lake\nopen Lake DSL\npackage «ReviewSubject» where\nrequire «ZipperGenFormal» from "formal-lean"\nlean_lib «Given» where\n  roots := #['+roots+']\n@[default_target]\nlean_lib «Solution» where\n  roots := #[`Solution]\nlean_lib «Check» where\n  roots := #[`Check]\n')
    (destination/'Solution.lean').write_text('import CompletionGoal\ntheorem submission : '+GOAL+' := by\n  sorry\n')
    if copy_library:
        shutil.copytree(PIN,destination/'formal-lean',ignore=shutil.ignore_patterns('.git','.lake'))
        if (PIN/'.lake/build').exists(): shutil.copytree(PIN/'.lake/build',destination/'formal-lean/.lake/build')
    else: (destination/'formal-lean').symlink_to(PIN,target_is_directory=True)
def prompt(k,arm,*,frozen=True):
    verify_size(k)
    return (source_root(k)/'prompts'/f'k{k}-{arm}.md').read_text()
