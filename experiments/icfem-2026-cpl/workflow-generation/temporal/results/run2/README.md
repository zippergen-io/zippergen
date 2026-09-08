# Status chain

`workflow.py` implements `specification.md` with deterministic local actions.
Three consecutive actions at L0 update `status`; only `token` is sent along
L0 → L1 → L2 → L3.

L3 evaluates `At[L0](since(Here.status != "failed", Here.status == "passed"))`.
This temporal guard checks L0's causally visible history: a pass must have
occurred with no subsequent failure. Checking preserves an existing pass.
ZipperGen carries the temporal monitor information through the chain without
adding application payloads or a workflow variable summarizing history.

Validate and test:

```sh
zippergen validate --json
python -m unittest -v
```

Run a preserved pass (returns `accept`):

```sh
zippergen run --input first_status=passed --input second_status=checking --input third_status=checking --input token=opaque
```

No model, connector, or service is required. Tests cover all 27 status
sequences with three opaque token values and verify the message chain.
