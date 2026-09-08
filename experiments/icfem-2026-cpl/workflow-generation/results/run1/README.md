# Coin chain

`workflow.py` implements `specification.md` using four lifelines and exactly
three application messages, forwarding the unchanged opaque token to `L3`.
The `L3`-owned Causal Past Logic guard `At[L0].outcome == "heads"` evaluates
the causally prior outcome at `L0`. ZipperGen supplies the guard's causal
coordination; the application token does not encode the outcome.

Inputs `outcome` (`heads` or `tails`) and `token` are strings owned by `L0`.
The returned `result` string belongs to `L3`. Local actions are deterministic;
no models, connectors, or deployment configuration are required.

```sh
zippergen validate --json
zippergen run --input outcome=heads --input token=opaque
zippergen run --input outcome=tails --input token=opaque
python3 -m unittest -v test_workflow.py
```

The tests check the input/output ownership, token-message chain, decision
owner, and both outcomes using the same token. CLI runs use ZipperGen's normal
local state directory for execution locking.
