# Local Deployment Booklet

This guide is for a developer who wants to run a simple ZipperGen system on one
local machine for a long time.

The recommended deployment shape is:

1. `zippergen deploy` creates a named, self-contained workflow deployment.
2. Normal settings are persisted in its profile and secrets in a private file.
3. The workflow uses one persistent SQLite file as its store.
4. Human approvals are completed through SQLite-backed adapters, such as the
   CLI or Telegram.
5. launchd on macOS or systemd on Linux keeps the workflow alive.
6. You inspect the system with `zippergen status`, `zippergen logs`, and
   `zippergen trace`.

ZipperChat can still be useful for visualization, but browser approval is
legacy. For deployed systems, SQLite is the source of truth.

`zippergen run` is useful for experiments. `zippergen serve` is a legacy,
low-level per-role command; neither is the normal long-running deployment path.

## The Big Idea

When a workflow runs, ZipperGen records the deployment-relevant history in
SQLite:

- messages between lifelines
- replay cursors
- snapshots
- LLM/action/effect journal results
- branch decisions
- human tasks
- human approval tokens
- notification adapter state
- trace events
- final workflow result, if the workflow terminates

If the process stops and starts again with the same SQLite store, ZipperGen
reconstructs the committed state and continues from there.

This does not mean every external-world byte must live in SQLite. For normal
text, prompts, approvals, draft bodies, and trace events, SQLite is fine. For
large attachments, PDFs, retrieved corpora, or long contexts, pass references
through the workflow instead:

```text
gmail:message:abc123
file:/Users/me/.zippergen/blobs/context-001.json
calendar:event:xyz789
```

SQLite should store the coordination state, durable IDs, and hashes. Large data
can live outside SQLite.

Internal workflow receives also use SQLite polling. A waiting role starts with a
short sleep for responsiveness and backs off up to one second while no matching
message is available. As soon as the role makes progress, the sleep resets to
the short value.

## The Commands You Will Use

Create and start a deployment:

```bash
uv run zippergen validate <module-or-path>:<workflow>
uv run zippergen show <module-or-path>:<workflow> --communications
uv run zippergen deploy <module-or-path>:<workflow>
```

The guided command collects declared settings, captures secrets without placing
them in the profile, installs declared packages into a managed environment,
runs one-time setup, checks readiness, and starts the platform user service.

Operate it by name:

```bash
uv run zippergen status <name>
uv run zippergen logs <name> --follow
uv run zippergen doctor <name>
uv run zippergen restart <name>
uv run zippergen configure <name> --restart
uv run zippergen deploy <name>       # snapshot and redeploy updated source
```

Approve a human task from the CLI:

```bash
uv run zippergen tasks --store <sqlite-store> --tokens
uv run zippergen approve --store <sqlite-store> --token <token>
uv run zippergen approve --store <sqlite-store> --token <token> --no
uv run zippergen approve --store <sqlite-store> --token <token> --value "edited text"
```

Run Telegram approvals:

```bash
uv run zippergen notify telegram --store <sqlite-store> --watch
```

The remaining manual sections explain the underlying pieces and are useful for
debugging or workflows without a deployment declaration. They are not the
recommended first-run procedure.

## Important Terms

Store:
The SQLite file for one deployed workflow run. Example:

```text
~/.zippergen/runs/command-center.sqlite
```

Workflow process:
The long-running `zippergen run ...` process.

Notifier process:
A separate process that watches the same SQLite store and delivers human tasks
to some channel. Today that can be `stdout` or Telegram.

Human task:
A durable approval/edit/input request stored in SQLite.

Token:
A durable external approval credential. Telegram and future adapters should use
tokens instead of raw task IDs.

Trace:
The recorded runtime events, useful for understanding what happened.

Replay:
Restarting from the committed SQLite history instead of starting from scratch.

## Safety Rules For A First Deployment

Follow these rules for the first real run:

1. Use one stable SQLite store path.
2. Start manually in terminals before using `launchd` or `systemd`.
3. Keep human approvals out-of-band through SQLite tasks.
4. Prefer creating email drafts over sending email automatically.
5. Avoid passing huge attachments or PDFs as workflow variables.
6. Check `status`, `tasks`, and `trace` often during the first run.
7. Restart once manually to confirm replay works.

Email drafts are acceptable for early deployment. A crash at the wrong moment can
create duplicate drafts, but duplicate drafts are usually manageable. Automatic
sending is different: duplicate sent emails are much more serious and should
have idempotency protection before being trusted.

## Part 1: Local Smoke Test

Do this before using the command center. It proves that the deployment loop
works on your machine:

- workflow starts
- SQLite store is created
- human task appears
- approval is completed through SQLite
- workflow resumes

### Terminal 1: Start The Workflow

```bash
uv run zippergen run examples/local_approval_deployment.py:local_approval \
  --store ~/.zippergen/runs/local-approval.sqlite \
  --input request="Create the Friday demo event" \
  --llm mock \
  --timeout 0
```

This command should keep running and wait for a human approval.

### Terminal 2: Inspect The Store

```bash
uv run zippergen status --store ~/.zippergen/runs/local-approval.sqlite
```

You should see a state like:

```text
State: waiting (waiting for 1 human task(s))
```

Now list the pending task:

```bash
uv run zippergen tasks --store ~/.zippergen/runs/local-approval.sqlite --tokens
```

Copy the token shown by the command.

Approve the task:

```bash
uv run zippergen approve \
  --store ~/.zippergen/runs/local-approval.sqlite \
  --token <token>
```

Terminal 1 should finish and print a JSON result.

### Inspect The Trace

```bash
uv run zippergen trace \
  --store ~/.zippergen/runs/local-approval.sqlite \
  --tail 50
```

This shows recent runtime events.

### Restart Test

Run the same workflow command again with the same store:

```bash
uv run zippergen run examples/local_approval_deployment.py:local_approval \
  --store ~/.zippergen/runs/local-approval.sqlite \
  --input request="Create the Friday demo event" \
  --llm mock \
  --timeout 0
```

Because the workflow already finished, it should restore the recorded result
rather than asking for the same approval again.

## Part 2: Telegram Approvals

Telegram is the first real out-of-band approval adapter.

The workflow process and Telegram process are separate. They talk only through
SQLite.

### Create A Telegram Bot

1. Open Telegram.
2. Message `@BotFather`.
3. Send `/newbot`.
4. Follow the prompts.
5. Copy the bot token.
6. Send any message to your new bot. This opens the chat.

Set the token:

```bash
export ZIPPERGEN_TELEGRAM_TOKEN=<bot-token>
```

Find your chat ID:

```bash
curl -s "https://api.telegram.org/bot$ZIPPERGEN_TELEGRAM_TOKEN/getUpdates"
```

Look for the message object and find:

```text
"chat":{"id":123456789,...}
```

Set it:

```bash
export ZIPPERGEN_TELEGRAM_CHAT_ID=123456789
```

### Start A Workflow

Use the local approval example again:

```bash
uv run zippergen run examples/local_approval_deployment.py:local_approval \
  --store ~/.zippergen/runs/local-approval-telegram.sqlite \
  --input request="Approve the Telegram deployment test" \
  --llm mock \
  --timeout 0
```

### Start The Telegram Notifier

In another terminal:

```bash
export ZIPPERGEN_TELEGRAM_TOKEN=<bot-token>
export ZIPPERGEN_TELEGRAM_CHAT_ID=<chat-id>

uv run zippergen notify telegram \
  --store ~/.zippergen/runs/local-approval-telegram.sqlite \
  --watch
```

You should receive a Telegram message with approval buttons.

For boolean tasks, press the Telegram button.

For text/edit tasks, reply:

```text
/zg <token> <your text>
```

The notifier records sent notifications in SQLite. If the notifier restarts, it
does not resend the same pending task by default. Use `--resend` only when you
intentionally want another copy.

## Part 3: Command Center Manual Deployment

Do this manually before using `launchd` or `systemd`.

### Decide On One Store Path

Use one stable path:

```bash
export ZG_STORE="$HOME/.zippergen/runs/command-center.sqlite"
```

Create the directory:

```bash
mkdir -p "$HOME/.zippergen/runs"
```

### Set Credentials

For OpenAI:

```bash
export OPENAI_API_KEY=<your-openai-key>
```

For Telegram:

```bash
export ZIPPERGEN_TELEGRAM_TOKEN=<bot-token>
export ZIPPERGEN_TELEGRAM_CHAT_ID=<chat-id>
```

For live Gmail and Calendar, run the setup commands used by the example:

```bash
uv run python examples/gmail_client.py --setup
uv run python examples/google_calendar_client.py --setup
```

### Terminal 1: Start Command Center

Hosted OpenAI:

```bash
uv run zippergen run examples/command_center.py:command_center \
  --store "$ZG_STORE" \
  --llm openai:gpt-4o \
  --services live \
  --timeout 0
```

Local Ollama:

```bash
uv run zippergen run examples/command_center.py:command_center \
  --store "$ZG_STORE" \
  --llm ollama:qwen2.5:7b \
  --services live \
  --llm-idle-timeout 300 \
  --timeout 0
```

`--llm-idle-timeout 300` means a managed local model can be released after five
minutes of inactivity.

### Terminal 2: Start Telegram Approvals

```bash
uv run zippergen notify telegram \
  --store "$ZG_STORE" \
  --watch
```

### Terminal 3: Observe

```bash
uv run zippergen status --store "$ZG_STORE"
uv run zippergen tasks --store "$ZG_STORE"
uv run zippergen trace --store "$ZG_STORE" --tail 50
```

During the first deployment, keep this terminal open.

### Manual Restart Test

Stop the workflow with `Ctrl-C`.

Start the exact same command again with the same store path.

Then check:

```bash
uv run zippergen status --store "$ZG_STORE"
uv run zippergen trace --store "$ZG_STORE" --tail 50
```

The workflow should continue from SQLite. It should not start from an empty
history.

## Part 4: Call Intake Manual Deployment

The call intake example watches a Gmail inbox for calls for projects, grants,
positions, fellowships, and similar opportunities. It only sends certified
senders to the LLM. Accepted messages are classified, converted to JSON, written
to a CSV table, and answered automatically by email with the extracted JSON.

The recommended setup is now one guided command:

```bash
uv run zippergen deploy examples/call_intake.py:call_intake
```

The workflow declaration tells ZipperGen to collect the LLM key, certified
senders, intake address, Gmail query, table destination, safe reply mode, OAuth
credential path, and polling/rate limits. ZipperGen then creates the managed
environment, installs the Google clients, performs missing Gmail and Sheets
OAuth setup, runs readiness checks, and starts launchd or systemd. The `export`
commands below are not needed for this guided path.

Operate it by name:

```bash
uv run zippergen status call-intake
uv run zippergen logs call-intake --follow
uv run zippergen restart call-intake
uv run zippergen configure call-intake --restart
```

The remainder of this part documents the equivalent manual configuration for
troubleshooting and older deployments.

Automatic sending has a built-in safeguard: the send effect will not send more
than 10 emails per hour. If the limit is reached, the workflow waits outside the
SQLite transaction until another send slot is available. Before creating a draft
or sending, the workflow also checks that the reply recipient is exactly the
parsed sender address from the incoming email. If the sender address is missing
or inconsistent, the response is refused.

### Files And State

Use stable paths:

```bash
export ZG_CALL_STORE="$HOME/.zippergen/runs/call-intake.sqlite"
export ZIPPERGEN_CALL_TABLE="$HOME/.zippergen/calls.csv"
export ZIPPERGEN_CALL_INTAKE_RESPONSE_LOG="$HOME/.zippergen/call-intake-responses.jsonl"
export ZIPPERGEN_CALL_SHEET_ID="<google-sheet-id>"
export ZIPPERGEN_CALL_SHEET_NAME="Calls"
export ZIPPERGEN_CALL_TABLE_TARGETS=both
export ZIPPERGEN_CALL_INTAKE_RECIPIENTS="zippergen.sandbox+calls@gmail.com"
export ZIPPERGEN_CALL_GMAIL_QUERY="is:unread in:inbox to:zippergen.sandbox+calls@gmail.com"
export ZIPPERGEN_CALL_INTAKE_SEND_MODE=send
export ZIPPERGEN_CALL_INTAKE_MAX_EMAILS_PER_HOUR=10
export ZIPPERGEN_CALL_INTAKE_POLL_SECONDS=60
mkdir -p "$HOME/.zippergen/runs"
```

The SQLite store is the deployment history. The JSONL response log is a
lightweight idempotency aid for replies and the source of the hourly send
counter. With `ZIPPERGEN_CALL_TABLE_TARGETS=both`, the Google Sheet is the
shareable user-facing table and the CSV remains a local backup.

### Certified Senders

Set exact email addresses or whole domains:

```bash
export ZIPPERGEN_CERTIFIED_SENDERS="alice@example.com,@trusted-lab.org"
```

Messages from other senders are marked processed without calling the LLM.

### Intake Address

If you use a Gmail plus address such as:

```text
zippergen.sandbox+calls@gmail.com
```

configure it in two places:

```bash
export ZIPPERGEN_CALL_INTAKE_RECIPIENTS="zippergen.sandbox+calls@gmail.com"
export ZIPPERGEN_CALL_GMAIL_QUERY="is:unread in:inbox to:zippergen.sandbox+calls@gmail.com"
```

The Gmail query is the first filter: the client fetches only unread inbox
messages sent to that address. The workflow then checks the message headers
again (`To`, `Cc`, `Delivered-To`, `X-Original-To`, and `Envelope-To`) before it
calls the LLM. If the configured intake address is not present, the message is
ignored.

### Google Sheet Table

Create a Google Sheet in the same Google account. Rename the tab to:

```text
Calls
```

Copy the spreadsheet ID from the URL:

```text
https://docs.google.com/spreadsheets/d/<spreadsheet-id>/edit
```

Then set:

```bash
export ZIPPERGEN_CALL_SHEET_ID="<spreadsheet-id>"
export ZIPPERGEN_CALL_SHEET_NAME="Calls"
export ZIPPERGEN_CALL_TABLE_TARGETS=both
export ZIPPERGEN_CALL_SHEETS_CREDENTIALS="$ZIPPERGEN_CALL_GMAIL_CREDENTIALS"
export ZIPPERGEN_CALL_SHEETS_TOKEN="$HOME/.zippergen/call-sheets-token.json"
```

Run the Sheets OAuth setup once:

```bash
uv run \
  --with google-auth \
  --with google-auth-oauthlib \
  --with google-api-python-client \
  python examples/call_intake_sheets_client.py --setup
```

The workflow owns the first row of the sheet and uses it as the table header. Do
not rename those columns. Share the sheet from Google Drive with view access for
readers. They do not need edit access.

### Gmail Setup

Run the call-intake Gmail OAuth setup:

```bash
uv run python examples/call_intake_email_client.py --setup
```

The client fetches unread mail without marking it read. The workflow marks a
message read only after it has been ignored, recorded, or replied to.

### Start Automatic Sending

```bash
export OPENAI_API_KEY=<your-openai-key>
export ZIPPERGEN_CALL_SHEET_ID="<google-sheet-id>"
export ZIPPERGEN_CALL_SHEET_NAME="Calls"
export ZIPPERGEN_CALL_TABLE_TARGETS=both
export ZIPPERGEN_CALL_INTAKE_RECIPIENTS="zippergen.sandbox+calls@gmail.com"
export ZIPPERGEN_CALL_GMAIL_QUERY="is:unread in:inbox to:zippergen.sandbox+calls@gmail.com"
export ZIPPERGEN_CALL_INTAKE_SEND_MODE=send
export ZIPPERGEN_CALL_INTAKE_MAX_EMAILS_PER_HOUR=10
export ZIPPERGEN_CALL_INTAKE_POLL_SECONDS=60

uv run zippergen run examples/call_intake.py:call_intake \
  --store "$ZG_CALL_STORE" \
  --llm openai:gpt-4o \
  --services live \
  --llm-idle-timeout 300 \
  --timeout 0
```

The workflow checks the inbox, handles all currently available work one message
at a time, and sleeps when no message is available. The default empty-inbox
sleep is 60 seconds. For very low-volume inboxes, 300 seconds is also reasonable:

```bash
export ZIPPERGEN_CALL_INTAKE_POLL_SECONDS=300
```

For a dry run, use drafts instead of sending:

```bash
export ZIPPERGEN_CALL_INTAKE_SEND_MODE=draft
```

`--llm-idle-timeout 300` matters for local models. It lets a managed local
backend release the model after five idle minutes. Hosted models such as OpenAI
do not hold local model memory.

For local Ollama:

```bash
uv run zippergen run examples/call_intake.py:call_intake \
  --store "$ZG_CALL_STORE" \
  --llm ollama:qwen2.5:7b \
  --services live \
  --llm-idle-timeout 300 \
  --timeout 0
```

Email is still an external side effect. The response log prevents ordinary
duplicates after a successful logged response, but it is not a perfect
transaction with Gmail. The hourly rate limit is enforced from successful
`mode=send` entries in the response log.

### Corrections

The response email contains the extracted JSON and a `call_id`. If the sender
finds an error, they can reply with corrected JSON or corrected fields. The
workflow treats that as a correction, keeps the `call_id`, and updates the CSV
row only if that `call_id` already exists.

This distinction matters:

- A new call with an existing `call_id` is treated as a duplicate. The workflow
  does not add a second row and does not modify the existing row. It emails the
  sender that the call is already recorded.
- A correction with an existing `call_id` updates the row.
- A correction with a missing `call_id` does not create a new row. It emails the
  sender that the call could not be found, so they can reply with the right
  `call_id` or send the message as a new call.

### Observe

```bash
uv run zippergen status --store "$ZG_CALL_STORE"
uv run zippergen trace --store "$ZG_CALL_STORE" --tail 50
cat "$ZIPPERGEN_CALL_TABLE"
```

## Part 5: Run Under macOS `launchd`

`zippergen deploy` and `zippergen start` now generate, install, and load the
launchd agent automatically. The manual template procedure below is retained
for troubleshooting custom installations.

Only do this after the manual deployment works.

Templates live here:

```text
deploy/launchd/com.zippergen.workflow.plist
deploy/launchd/com.zippergen.telegram-notifier.plist
```

### Step 1: Copy The Templates

```bash
mkdir -p ~/Library/LaunchAgents ~/.zippergen/logs

cp deploy/launchd/com.zippergen.workflow.plist \
  ~/Library/LaunchAgents/com.zippergen.workflow.plist

cp deploy/launchd/com.zippergen.telegram-notifier.plist \
  ~/Library/LaunchAgents/com.zippergen.telegram-notifier.plist
```

### Step 2: Edit The Files

Open both files and replace:

```text
/Users/YOU/path/to/zippergen
/Users/YOU/.zippergen/runs/command-center.sqlite
REPLACE_ME
```

Use absolute paths. `launchd` does not load your ordinary shell environment.

If `launchd` cannot find `uv`, replace:

```text
/usr/bin/env
uv
```

with the absolute path to `uv`. Find it with:

```bash
which uv
```

### Step 3: Start The Services

```bash
launchctl bootstrap "gui/$(id -u)" \
  ~/Library/LaunchAgents/com.zippergen.workflow.plist

launchctl bootstrap "gui/$(id -u)" \
  ~/Library/LaunchAgents/com.zippergen.telegram-notifier.plist
```

### Step 4: Check Status

```bash
launchctl print "gui/$(id -u)/com.zippergen.workflow"
launchctl print "gui/$(id -u)/com.zippergen.telegram-notifier"
```

Check logs:

```bash
tail -f ~/.zippergen/logs/workflow.err.log
tail -f ~/.zippergen/logs/telegram-notifier.err.log
```

Check ZipperGen state:

```bash
uv run zippergen status --store ~/.zippergen/runs/command-center.sqlite
```

### Stop The Services

```bash
launchctl bootout "gui/$(id -u)" \
  ~/Library/LaunchAgents/com.zippergen.workflow.plist

launchctl bootout "gui/$(id -u)" \
  ~/Library/LaunchAgents/com.zippergen.telegram-notifier.plist
```

## Part 6: Run Under Linux `systemd --user`

`zippergen deploy` and `zippergen start` now generate, install, and control the
systemd user unit automatically. The manual template procedure below is
retained for troubleshooting custom installations.

Only do this after the manual deployment works.

Templates live here:

```text
deploy/systemd/zippergen-workflow.service
deploy/systemd/zippergen-telegram-notifier.service
```

### Step 1: Copy The Templates

```bash
mkdir -p ~/.config/systemd/user ~/.zippergen/runs

cp deploy/systemd/zippergen-workflow.service \
  ~/.config/systemd/user/zippergen-workflow.service

cp deploy/systemd/zippergen-telegram-notifier.service \
  ~/.config/systemd/user/zippergen-telegram-notifier.service
```

### Step 2: Edit The Files

Replace:

```text
%h/path/to/zippergen
REPLACE_ME
```

Make sure the `ExecStart` commands contain the exact workflow and store path you
tested manually.

### Step 3: Start The Services

```bash
systemctl --user daemon-reload
systemctl --user enable --now zippergen-workflow.service
systemctl --user enable --now zippergen-telegram-notifier.service
```

### Step 4: Check Status And Logs

```bash
systemctl --user status zippergen-workflow.service
systemctl --user status zippergen-telegram-notifier.service

journalctl --user -u zippergen-workflow.service -f
journalctl --user -u zippergen-telegram-notifier.service -f
```

Check ZipperGen state:

```bash
uv run zippergen status --store ~/.zippergen/runs/command-center.sqlite
```

### Stop The Services

```bash
systemctl --user stop zippergen-workflow.service
systemctl --user stop zippergen-telegram-notifier.service
```

## Part 7: What Happens After A Crash

On restart with the same SQLite store, ZipperGen does not simply re-execute
everything.

It reads the committed history and reconstructs the workflow state:

- recorded messages are replayed from SQLite
- completed human tasks are reused
- LLM results are reused if already journaled
- branch decisions are reused if already journaled
- effect results are reused if already journaled
- pure actions may be recomputed because they are assumed deterministic

The risky window is:

```text
external effect succeeds
process crashes before SQLite records the effect result
```

In that case, ZipperGen cannot know the effect succeeded, so it may run the
effect again.

For Gmail drafts, this can create duplicate drafts. That is acceptable for an
early local deployment. For automatically sent email, payments, or destructive
operations, add idempotency protection before trusting the system.

## Part 8: Effects And Idempotency

An effect is a Python action that touches the outside world.

Examples:

- create Gmail draft
- send email
- create calendar event
- delete calendar event
- post to Slack
- write a file

Safer effects have stable IDs or idempotency keys. For example:

```text
action = "create_draft"
input_hash = "abc123"
external_marker = "zippergen:create_draft:abc123"
```

Then the effect can check whether it already ran before doing it again.

For the first deployment, this rule is enough:

```text
Drafts are okay.
Automatic sends and destructive effects need more care.
```

## Part 9: Large Data

SQLite is fine for normal coordination data:

- email text
- approval context
- draft text
- LLM outputs
- trace events

Avoid putting huge data directly in workflow variables:

- attachments
- PDFs
- long documents
- large RAG contexts
- binary files

Pass references instead:

```text
file:/Users/me/.zippergen/blobs/doc-001.pdf
gmail:message:abc123
s3://bucket/key
vector-store:item:xyz
```

If the reference matters for replay, store a hash too.

## Part 10: Common Problems

### The Workflow Does Nothing

Check status:

```bash
uv run zippergen status --store <store>
```

Check pending tasks:

```bash
uv run zippergen tasks --store <store>
```

Check recent trace:

```bash
uv run zippergen trace --store <store> --tail 50
```

### Telegram Sends Nothing

Check:

```bash
echo "$ZIPPERGEN_TELEGRAM_TOKEN"
echo "$ZIPPERGEN_TELEGRAM_CHAT_ID"
```

Make sure there is a pending task:

```bash
uv run zippergen tasks --store <store>
```

Run one notifier pass without `--watch`:

```bash
uv run zippergen notify telegram --store <store>
```

Use `--resend` only if a task was already notified and you need another copy:

```bash
uv run zippergen notify telegram --store <store> --resend
```

### The Store Path Is Wrong

If the workflow and notifier use different SQLite files, they will not see each
other.

Use the same absolute path in every command:

```bash
export ZG_STORE="$HOME/.zippergen/runs/command-center.sqlite"
```

Then use:

```bash
--store "$ZG_STORE"
```

### `launchd` Cannot Find `uv`

Use an absolute path. Find it with:

```bash
which uv
```

Then edit the plist.

### The Workflow Repeats Something After Restart

Check whether the thing that repeated was already committed to SQLite:

```bash
uv run zippergen trace --store <store> --tail 100
```

If an external effect succeeded but crashed before the SQLite journal commit,
the effect may run again. This is why early deployments should prefer drafts and
non-destructive actions.

## Part 11: First Deployment Checklist

Before using an OS supervisor:

- [ ] `uv run zippergen run ... --timeout 0` works manually.
- [ ] `uv run zippergen notify telegram ... --watch` works manually.
- [ ] `zippergen status` shows the expected store.
- [ ] `zippergen tasks` shows pending human tasks.
- [ ] Telegram approval completes a task.
- [ ] `zippergen trace --tail 50` shows recent events.
- [ ] Stopping and restarting the workflow uses the same store.
- [ ] You know where logs will go.
- [ ] You understand that duplicate drafts can happen after a crash.
- [ ] You are not automatically sending email without idempotency protection.

After that, move the exact same commands into `launchd` or `systemd`.

## Recommended First Real Run

Use hosted OpenAI first because it removes local model memory management from
the first deployment:

```bash
export ZG_STORE="$HOME/.zippergen/runs/command-center.sqlite"
export OPENAI_API_KEY=<your-openai-key>
export ZIPPERGEN_TELEGRAM_TOKEN=<bot-token>
export ZIPPERGEN_TELEGRAM_CHAT_ID=<chat-id>

uv run zippergen run examples/command_center.py:command_center \
  --store "$ZG_STORE" \
  --llm openai:gpt-4o \
  --services live \
  --timeout 0
```

In another terminal:

```bash
uv run zippergen notify telegram \
  --store "$ZG_STORE" \
  --watch
```

In a third terminal:

```bash
uv run zippergen status --store "$ZG_STORE"
uv run zippergen tasks --store "$ZG_STORE"
uv run zippergen trace --store "$ZG_STORE" --tail 50
```

Once this works, move to `launchd` or `systemd`.
