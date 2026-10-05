# Observing native CLI progress

`make_cli_assistant_backend(..., on_progress=callback)` adds an optional public
progress observer. Leaving it unset preserves buffered execution. It does not
change action declarations, workflow messages, permissions, prompt construction,
result validation or persistent conversation ownership.

```python
backend = make_cli_assistant_backend("codex", on_progress=observe)
```

Each invocation first calls the observer with `kind="invocation.started"` and
`invocation_id`, `participant`, `endpoint`, `action`, and `backend`. This small
registration callback runs synchronously and must return promptly. Subsequent
public tool events include that identity and run on an observer thread. Consumers
must be thread-safe and reject progress after an invocation has ended.

Tool records expose `kind` (`command` or `file`), `phase` (`started` or `completed`),
and `tool_id` where available. Commands include their command string and observed
exit code, plus optional public `output` on completion. File reads may carry
`offset` and `limit`; other files include operation/path and status where available. Embeddings
should classify these into short UI labels rather than display raw arguments.
They must not treat progress, a successful test command, or missing progress as
an action result, review verdict, protocol message or semantic state transition.

Codex uses its existing `--json` stream. Claude switches only its output format to
`stream-json --verbose`, then extracts the final result envelope for the existing
result and session validators. Claude tool-use blocks report started tools; corresponding `tool_result` blocks
report completion, correlated by tool-use ID within one invocation. `is_error`
is reported as status, never fabricated into a numeric exit code. This adapter
does not infer completion from silence. See the official
[Claude programmatic CLI documentation](https://code.claude.com/docs/en/headless).
Public command results are optional and bounded to 262,144 characters per event;
`output_truncated` explicitly marks a longer result. Embeddings must keep arbitrary
result text out of short UI labels and handle it with the same care as other local
trace payloads. File-read bodies remain excluded.
Neither adapter forwards reasoning, assistant text, patches, arbitrary
input objects, token deltas or unknown event types through the observer.

Both output pipes are drained concurrently and the input writer cannot deadlock
against them. A bounded 64-event queue isolates tool telemetry from slow or failed
observers; observer exceptions are ignored. The CLI timeout and final result are
independent of observation. Some telemetry may be dropped under load or at process
exit, so this is a best-effort progress surface, not an audit log. The final
validated action output and runtime trace remain authoritative. Late callbacks
carry their original invocation identity.

CLI arguments, environment isolation, read/write sandbox flags, native session
IDs/resumes and per-participant locks remain owned by the native backend.
Observers survive new execution construction and durable participant binding.
