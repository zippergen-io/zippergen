import threading
from zippergen import Lifeline, Var, workflow
from zippergen.actions import llm as llm_deco, pure
from zippergen.projection import project
from zippergen.store import open_store
from zippergen.serve import run_role

A = Lifeline("A"); B = Lifeline("B")
n = Var("n", int, default=0); m = Var("m", int, default=0)
label = Var("label", int, default=0); got = Var("got", int, default=0)

CALLS = {"n": 0}
def counting_backend(action, inputs):
    CALLS["n"] += 1
    return {"label": CALLS["n"] * 10}          # non-deterministic across calls

# Match the exact @llm signature used in examples/diagnosis.py and tests/ when
# writing this fixture; `parse` and `outputs` keyword shapes come from actions.py.
classify = llm_deco(system="s", user="{m}", parse="json", outputs=[("label", int)])
@classify
def classify_fn(m: int) -> int: ...

@pure
def relay(m: int) -> int:
    return m

@workflow
def one_round(n: int @ A):
    A(n) >> B(m)
    B: label = classify_fn(m)
    B(label) >> A(got)
    return got @ A

def _run(path, role, local, seed):
    return run_role(open_store(path), role, local, dict(seed), one_round.ns,
                    llm_backend=counting_backend)

def test_external_act_memoized_across_restart(tmp_path):
    path = str(tmp_path / "s.sqlite")
    la, lb = project(one_round, A), project(one_round, B)
    envs = {}
    tb = threading.Thread(target=lambda: envs.__setitem__("B", _run(path, "B", lb, {})))
    tb.start()
    envs["A"] = _run(path, "A", la, {"n": 1})
    tb.join(timeout=15)
    first_calls = CALLS["n"]
    assert first_calls == 1 and envs["A"]["got"] == 10
    # Re-run B from the committed log: the classify LLM must NOT be called again.
    env_b2 = _run(path, "B", lb, {})
    assert CALLS["n"] == first_calls               # memoized, no re-invocation
    assert env_b2["label"] == 10


def test_default_seed_inputs_overlay():
    from zippergen.serve import _seed_inputs
    merged = _seed_inputs(one_round, {"n": 3})
    assert merged["n"] == 3          # caller input wins
    assert merged["m"] == 0          # Var default carried through when not supplied
