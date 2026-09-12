# Two workers sharing a work pool

Producer submits two independent JSON jobs. It notifies both workers only
after the submissions have finished. WorkerA and WorkerB then each attempt to
claim one available job without waiting. A worker reports an empty result if
no job is available; otherwise it computes a short result and acknowledges
the claim. Producer collects the two reports and returns them.

The pool selects jobs in ready-queue order. Which worker receives which job is
unspecified, and their processing may finish in either order. Neither worker
may acknowledge the other's claim. Jobs and receipts survive a durable resume;
retrying a submission or claim must not create a second job or select a second
item. Acknowledgement marks work complete, while release or lease expiry puts
unfinished work at the back of the ready queue. The default lease is 300
seconds and is not renewed automatically.

The pool belongs to this execution's managed SQLite store. A new execution has
a new pool. Pool operations are external effects relative to the interpreter
transaction and do not transfer CPL causal context. The explicit notification
messages establish that submission precedes each worker's attempt.

Run from the repository root using the checkout's environment (the installed
PyPI version may not yet include `Pool`):

```bash
uv sync
.venv/bin/python -m zippergen.serve validate examples/work_pool/workflow.py:work_pool
.venv/bin/python -m zippergen.serve run --workflow examples/work_pool/workflow.py:work_pool
```

Processing in this example is deterministic. An application that performs
external work before acknowledgement must make that work safe to repeat,
using the returned job ID when the external service supports idempotency.
