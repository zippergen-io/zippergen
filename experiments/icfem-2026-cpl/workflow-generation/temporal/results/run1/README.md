# Status chain

The self-contained `status_chain` workflow records three status updates at L0,
then sends an opaque token through L1 and L2 to L3. L3 evaluates the causal
history of L0's recorded `status` using `At[L0](since(...))`: a pass remains
valid until a failure, and another pass restores it. No application variable
summarizes that history, and the application messages contain only the token.

Validate and run the exhaustive offline tests:

```sh
zippergen validate --json
python -m unittest -v
```

Run an example that returns `accept`:

```sh
zippergen run --input first_status=passed --input second_status=checking --input third_status=checking --input token=opaque
```

All actions are deterministic and local; no model, connector, or service setup
is needed.
