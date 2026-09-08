# Coin chain

`workflow.py` implements `coin_chain` with the four participants and three
token transfers in `specification.md`. Inputs `outcome` and `token` belong to
`L0`; `outcome` must be `heads` or `tails`.

The decision belongs to `L3`. Its structural causal-past guard,
`(At[L0].outcome == "heads") @ L3`, uses ZipperGen's causal-past monitoring
along the chain. The application token stays unchanged and does not encode
the outcome. This relies on framework monitoring, not inference from the
opaque token; the outcome is not hidden from that monitoring mechanism.
Both result actions are deterministic and require no models or services.

```sh
zippergen validate --json
zippergen run --input outcome=heads --input token=opaque
zippergen run --input outcome=tails --input token=opaque
python3 -m unittest -v test_workflow.py
```

Tests check the protocol and both outcomes with identical tokens, including
tokens that themselves spell `heads` or `tails` and the empty string.
