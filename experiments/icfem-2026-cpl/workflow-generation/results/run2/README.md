# Coin chain

`workflow.py` implements `specification.md` with deterministic local actions.
The three application messages forward the token unchanged. At L3, the
causal-past guard `At[L0].outcome == "heads"` observes L0's state through the
message chain; ZipperGen maintains the causal information separately from the
application token. L3 owns the decision and returns the corresponding string.
As specified, `outcome` must be `heads` or `tails`.

No models, credentials, connectors, or services are required.

```sh
zippergen validate --json
zippergen run --input outcome=heads --input token=opaque-token
python3 -m unittest -v
```

The tests check the application-message chain, decision ownership, and both
outcomes using identical tokens, including misleading and empty tokens.
