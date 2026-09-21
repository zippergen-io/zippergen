# Waiting for external data

A short polling effect can leave its continuation parked in the SQLite runner:

```python
from zippergen import EffectPending, Json, effect

@effect(visible=False)
def next_message(thread_id: str, after_round: int) -> Json:
    message = mailbox.peek(thread_id, after_round)
    if message is None:
        raise EffectPending(retry_after=1)
    return message
```

`mailbox.peek` stands for a bounded application read. The workflow invokes this
as one ordinary effect. It needs no loop just to represent an empty poll.

Raising `EffectPending` produces no output and does not advance or checkpoint
the action. The runner tries another enabled session or parallel branch at that
participant. It probes again after the positive finite delay. Reads are still
polling, so choose an interval appropriate to the external service.

The saved control position is the recovery record. Retry deadlines are only
in-memory scheduling hints. After restart the runner may probe immediately.
Once a result arrives, the output and continuation commit together using the
usual effect semantics. A crash before that commit can repeat the operation.

Only use this for a retry-safe probe. Do not use it to hide an uncertain send,
an authentication error or a programming defect. A successful claim must stay
redeliverable until the workflow has saved what it needs. Keep correlation
inputs session-private if another session could overwrite them while waiting.
Each probe reads its current inputs again.

`visible=False` avoids recording routine start/wait events for every probe.
With default visibility, history records the attempts and their waiting state.
Even quiet polling still performs reads and can update diagnostic status when
that status changes. No claim of zero CPU use or disk I/O is implied.

This experimental facility uses the SQLite runner, including `run_sqlite`,
`zg run --durable` and deployments. Disposable runs using the in-memory runner
and the Python `run` API reject it.
Calls must return or raise promptly. This is not an asynchronous executor for
long model calls or arbitrary blocking effects. Human approvals continue to
use `@human` and its task queue.
