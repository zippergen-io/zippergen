# Concurrent draft approvals

Mailbox selects three fixed example requests. Each request starts its own
session with the same Mailbox, Writer and User participants. Intake continues
while earlier requests wait for approval.

Writer prepares a fixed draft. User approves or rejects it. Mailbox records
approved drafts in a shared local list. Rejected drafts are not recorded.
This example sends no email and uses no external model or credentials.

Each session has its own request, draft and approval cells. Mailbox copies
the selected request at its local fork. Other participants receive the request
through normal session messages. The completed list is intentionally inherited
by all Mailbox sessions. There is no shared memory between participants.
Validation therefore reports an expected shared-write warning for
`Mailbox.completed`. The request, draft and approval fields remain private.

With durable execution, all three approvals can be pending together and can
be answered in any order. Restart resumes those same tasks and session cells.
The parent finishing intake does not cancel the children. The finite run
finishes after all local session continuations have finished.

From this directory, using a checkout with experimental spawn support:

```sh
zg validate workflow.py:draft_sessions
zg show workflow.py:draft_sessions --detail full
zg run --workflow workflow.py:draft_sessions --durable
```

In another terminal in the same directory:

```sh
zg run tasks
zg run approve --task TASK_ID --yes
zg run approve --task ANOTHER_TASK_ID --no
```

Choose task IDs from `zg run tasks`. To resume after stopping the run, use
`zg run --resume`. A plain run uses terminal prompts and does not demonstrate
several pending approvals. This is experimental syntax, not part of a published
release yet.
