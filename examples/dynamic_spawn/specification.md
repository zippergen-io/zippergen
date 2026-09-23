# Dynamically assigned access reviews

Manager dispatches two requests without waiting for either review to finish.
Each request creates an AccessReview session. Manager and IT retain their
endpoints, while Reviewer is assigned to reviewer-east or reviewer-west using
a complete Manager-owned binding. These reviewer endpoints have no program in
the parent session. All endpoints are hosted by the existing local runtime;
this example does not demonstrate remote deployment.

At each retained role's local fork, fresh-all copies its child-relevant fields.
Reviewer starts with uninitialized child-local protocol fields and learns the
request through Manager's message. Reviewer owns the approval choice and sends
the generated decision to the child roles that need it. IT provisions only the
approved request. The fake policy approves read:alpha and rejects write:beta.
The actions are deterministic local functions with no external services.

The two visits to the same spawn site must create distinct sessions, isolate
request/decision state, preserve the selected reviewers, and let the parent
continue. Parent completion does not stop children. The runner returns only
after outstanding child work finishes. The returned integer counts dispatches,
not approvals or child results; no implicit return channel is provided.

Run from the repository root:

```sh
python examples/dynamic_spawn/workflow.py
zg validate examples/dynamic_spawn/workflow.py:access_reviews
zg show examples/dynamic_spawn/workflow.py:access_reviews
zg show --agent Reviewer examples/dynamic_spawn/workflow.py:access_reviews
```

The reviewer endpoint names are explicit demo configuration passed to the
runner. Production selection would use configured endpoints and a separately
implemented discovery action; it would retain that action's ordinary effect
replay guarantees.
