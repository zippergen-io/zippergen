# Status chain

`workflow.py` implements `specification.md` with deterministic local actions.
L0 records three updates to `status`, then forwards the opaque token through
L1 and L2 to L3. L3's causal-past `since` guard accepts when a passed update
has occurred without a subsequent failed update. Checking preserves that pass.
The guard references only `status`; no application history summary is stored
or sent. ZipperGen propagates the temporal monitor through the message chain.

Validate:

```sh
zippergen validate --json
```

Run a preserved pass:

```sh
zippergen run --input first_status=passed --input second_status=checking --input third_status=checking --input token=opaque
```

Run all 27 possible status histories (no model or external service needed):

```sh
python3 -m unittest -v test_workflow.py
```
