# Local Deployment

This is the current recommended shape for a simple local ZipperGen deployment:

1. Run the workflow through one persistent SQLite store.
2. Run one or more notification adapters against the same store.
3. Let the operating system supervise both processes.
4. Inspect the store with `status`, `tasks`, and `trace`.

SQLite is the coordination boundary. The workflow process, CLI approval tools,
Telegram adapter, and future adapters all communicate through the same store.

## Local Smoke Test

```bash
uv run zippergen run examples/local_approval_deployment.py:local_approval \
  --store ~/.zippergen/runs/local-approval.sqlite \
  --input request="Create the Friday demo event" \
  --llm mock \
  --timeout 0
```

In another terminal, inspect and approve through the CLI:

```bash
uv run zippergen status --store ~/.zippergen/runs/local-approval.sqlite
uv run zippergen tasks --store ~/.zippergen/runs/local-approval.sqlite --tokens
uv run zippergen approve --store ~/.zippergen/runs/local-approval.sqlite --token <token>
uv run zippergen trace --store ~/.zippergen/runs/local-approval.sqlite --tail 50
```

`--timeout 0` means the runner has no deadline. Use it for workflows that are
meant to keep running under `launchd` or `systemd`.

## Telegram Approvals

Create a bot with `@BotFather`, send your bot one message, then set:

```bash
export ZIPPERGEN_TELEGRAM_TOKEN=<bot-token>
export ZIPPERGEN_TELEGRAM_CHAT_ID=<your-chat-id>
```

If you do not know the chat id yet, temporarily run:

```bash
curl -s "https://api.telegram.org/bot$ZIPPERGEN_TELEGRAM_TOKEN/getUpdates"
```

Then start the adapter:

```bash
uv run zippergen notify telegram \
  --store ~/.zippergen/runs/local-approval.sqlite \
  --watch
```

Boolean tasks get Telegram buttons. Text/edit tasks can be completed by sending:

```text
/zg <token> <your text>
```

The adapter records sent notifications in SQLite, so restarting it does not
resend the same pending task by default. Use `--resend` when you intentionally
want another copy.

## Command Center

For command center with live services and hosted OpenAI:

```bash
uv run zippergen run examples/command_center.py:command_center \
  --store ~/.zippergen/runs/command-center.sqlite \
  --llm openai:gpt-4o \
  --services live \
  --llm-idle-timeout 300 \
  --timeout 0
```

Then run Telegram approvals:

```bash
uv run zippergen notify telegram \
  --store ~/.zippergen/runs/command-center.sqlite \
  --watch
```

For local models, use an Ollama spec such as `--llm ollama:qwen2.5:7b`. The
idle timeout releases the model after inactivity.

## macOS `launchd`

Copy the templates in `deploy/launchd/`, replace the placeholder paths and
environment values, create the log directory, then load them:

```bash
mkdir -p ~/.zippergen/logs ~/Library/LaunchAgents
launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/com.zippergen.workflow.plist
launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/com.zippergen.telegram-notifier.plist
```

Use absolute paths for `uv` if `launchd` cannot find it.

## Linux `systemd --user`

Copy the templates in `deploy/systemd/` to `~/.config/systemd/user/`, edit paths
and environment values, then run:

```bash
systemctl --user daemon-reload
systemctl --user enable --now zippergen-workflow.service
systemctl --user enable --now zippergen-telegram-notifier.service
```

## Remaining External Effect Rule

ZipperGen journals effect results after the Python effect returns. If the
process crashes after a real external side effect but before the journal commit,
the effect may run again on restart. Effects that talk to external systems
should therefore use an idempotency key when the external system supports one,
or check local/external state before repeating the side effect. The
`examples/local_approval_deployment.py` example uses this pattern for its local
audit file.
