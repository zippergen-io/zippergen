# Approval context carried through a pool

Producer receives a Boolean approval decision for request `report-17`. It
records the request and decision locally, then submits the request to `jobs`.
Worker polls for a claim, waiting briefly between empty attempts. There is no
message from Producer to Worker: the successful claim is the causal handoff.

Worker processes the request only when CPL establishes that the latest
causally visible Producer state approved the same request and returned the
same job ID. Otherwise it reports rejection. Both processing and rejection
are terminal handling outcomes, so Worker acknowledges either one.

The pool transfers the producer's context at the completed put, including its
returned job ID. Empty claims transfer nothing. The guard checks causal
evidence, not Producer's current wall-clock state; a later approval revocation
would need its own coordination. Request and job correlation are explicit.
`At` is a latest-visible view, so another later message or job can supersede
the stored view; it is not a per-job archive of approvals.

Jobs, causal metadata and receipts survive a durable resume. A crash between
the pool commit and the role checkpoint replays the same result and incoming
context, counting the action once. The example uses the default fixed
300-second lease and no external processing service.

From the repository root, after `uv sync`:

```bash
.venv/bin/python -m zippergen.serve validate examples/work_pool_cpl/workflow.py:approved_work_pool
.venv/bin/python -m zippergen.serve run --workflow examples/work_pool_cpl/workflow.py:approved_work_pool --input allow=true
.venv/bin/python -m zippergen.serve run --workflow examples/work_pool_cpl/workflow.py:approved_work_pool --input allow=false
```

Expected results are `processed report-17` and `rejected report-17`.
