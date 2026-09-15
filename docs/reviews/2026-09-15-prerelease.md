# Review before the next alpha release

Reviewed on 15 September 2026, starting from `8a2a645`.

The review covered deployment preparation and publication, start/stop/reset,
SQLite role recovery and message consumption, workflow compatibility checks,
pool receipts and leases, Google authorization and connectors, private files,
and Telegram task routing. This is a focused source review with regression
tests. It is not a full security audit or a proof of the runtime.

## Confirmed findings and fixes

### Google and Telegram error reports could expose private data

Gmail and Sheets included raw HTTP response bodies in exceptions. Google OAuth
and Telegram also chained upstream errors into tracebacks. Response bodies and
transport exceptions can include private content or request details. The runtime
copies exception messages into durable failure records, so the issue extends
beyond terminal output.

The affected HTTP and OAuth paths now use bounded, fixed explanations and retain
HTTP status codes without copying response bodies. Upstream exception chains are
suppressed in ordinary tracebacks. Gmail and Sheets share the HTTP error handling.
Calendar already sanitized its own HTTP errors and benefits from the shared
OAuth fix. Telegram also rejects a non-object JSON response with its named API
error instead of an incidental attribute error.

Separately, default dataclass representations printed Google credentials in
Gmail, Sheets, Calendar and authorization handoff objects. Credential fields now
use `repr=False`. This prevents accidental printing, not deliberate access to
the credential field or introspection by a debugger.

Regressions inject synthetic private strings into HTTP bodies, transport errors,
OAuth failures and credential fields. They check exception messages, formatted
tracebacks and object representations. These tests failed before the fixes.
The fixes do not remove anything already stored in old logs or failure records.

Tests: [Google error privacy](../../tests/test_google_error_privacy.py) and
[Telegram notifications](../../tests/test_telegram_notify.py).

### A malformed Sheets response could change the row being updated

`read_rows` silently skipped entries that were not lists. `upsert_row` then
enumerated the shortened result to compute physical spreadsheet row numbers.
A malformed entry before a matching key could therefore cause a write to the
wrong row. Falsey malformed collections were also interpreted as an empty sheet.

The connector now rejects malformed collections and entries before writing.
Valid empty rows remain in the result, so their physical positions are retained.
Tests cover rejection without writes and an update after a legitimate blank row.

Tests: [Google Sheets](../../tests/test_google_sheets.py).

### Sheets replacement cleared data before validating its replacement

`replace_rows` cleared the remote range before converting and serializing the
new values. An invalid value could leave the existing sheet empty even though
the replacement was invalid before any network call.

The complete payload is now prepared and checked before clearing. A regression
passes an unserializable value and checks that no remote request is made.
This does not make the two remote requests atomic.

Tests: [Google Sheets](../../tests/test_google_sheets.py).

### Two tests used the real site directory

The first full run had two failures because a CLI test and a manifest test
attempted to write under the user's actual ZipperGen home. Both now select a
temporary site directory. They no longer depend on permissions or configuration
outside the test environment.

Tests: [availability](../../tests/test_availability.py) and
[manifest shape](../../tests/test_manifest_shape.py).

## Existing behavior checked

- Redeployment reuses the existing store and checks workflow compatibility
  before publishing a candidate. Starting uses the published deployment.
  Reset archives the active store and leaves the service stopped.
- Sending a durable message and advancing its sender commit together.
  Receiving deletes the queued message in the same transaction that advances
  the receiving role. Recovery reads saved variables, control and monitor state.
- External calls run outside the SQLite write transaction. Their results and
  successor state commit together after the call returns.
- Pool effects record the request and result together. Repeating an invocation
  returns its receipt. A repeated claim does not renew its lease, and expired
  tokens cannot acknowledge a reassigned job. CPL recovery uses saved context.
- Telegram routing establishes task ownership before applying the local
  approver policy. Completing a task and consuming its response token are
  transactional.
- Deployment source, Python environment and credential generations are prepared
  before publication. Existing tests cover failed preparation and recovery.

No additional defect was confirmed in these reviewed paths. That conclusion is
limited to the code inspected and the test scenarios exercised.

## Limits that remain relevant to release

- An external effect may run again if it succeeds before its result is committed.
  Gmail sends and draft creation do not have an exactly-once guarantee.
- Sheets replacement still clears and writes in separate requests. A network
  failure between them can leave an empty table. Keyed upsert is a read followed
  by a write and requires a single writer. Concurrent writers can append
  duplicate keys. The connector docstrings now state these limits explicitly.
- Calendar request IDs recover an earlier creation. Availability reads do not
  reserve a timeslot, and notification delivery is not exactly-once.
- The workflow compatibility fingerprint protects stored control positions and
  value shapes. It deliberately excludes action bodies, prompts and guard
  computations. Passing this check is not proof that a code change preserves
  business behavior.
- Private credential files use filesystem permissions. This review does not
  establish encryption at rest or assess the security of the deployment host.
- The live Linux systemd integration test must still run on an appropriate
  Linux host before release. No production service was changed in this review.

## Validation

- Full framework suite on Python 3.12: **1,465 passed, 2 skipped**. The skipped
  checks are the Linux-only systemd lifecycle test and the local stale-build-tree
  check, since there is no `build/lib` tree in this checkout.
- Pyright: **0 errors, 0 warnings**.
- Source distribution and wheel built successfully. `twine check --strict`
  passed for both. The wheel's Python module inventory matches the source tree.
- The wheel installed into a clean temporary environment without dependencies.
  `zg --help`, `zg deploy --help`, `zg skill` and validation of the email approval
  example passed. The Google connector modules and shared HTTP helper import
  without the optional Google libraries. Actual Google authorization still
  requires the `google` extra.
- Regression tests use fake services and synthetic secrets. No live Google
  write, Telegram message or production deployment was performed.

Python 3.11/3.13 and the real Linux service manager still belong to the release
validation matrix. The test artifacts retain the current `0.1.0a3` version and
are for local verification only, not upload to PyPI.

The changes do not alter workflow syntax, deployment command meanings or the
SQLite schema. Existing saved executions do not need a reset for these fixes.
The package version remains unchanged. No release has been published.
