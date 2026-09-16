# Security boundaries

ZipperGen runs trusted Python workflows on infrastructure controlled by the
operator. Importing a workflow can execute Python. Deployment setup commands
and dependencies must also be trusted. Validation and projection do not make
arbitrary Python safe to execute.

## Coding assistants and credentials

ZipperGen filters the environment passed to Codex and Claude. Workflow model
keys and connector credentials are excluded. The CLI keeps its own login
locations and selected process settings, including proxy settings. Those
settings can themselves contain credentials.

This filter does not isolate the filesystem. Assistant processes run under the
workflow's operating-system account. Their working directory is not a boundary
against reading other files. Read-only execution restricts changes, but does
not establish that private credential files are unreadable.

An email, document or other workflow input can contain instructions intended
to manipulate an assistant. Prompt instructions alone do not protect secrets
from such input. Stronger separation requires an assistant execution service,
separate account or container that cannot read the workflow's credentials.
ZipperGen does not automatically provide that separation. Giving the whole
workflow a dedicated account protects other accounts, but does not separate
its assistant from its connectors.

The default assistant options restrict external tools and shell capabilities.
Actual enforcement depends on the selected CLI and host. `zg assistant check`
checks for the required options. It does not prove sandbox enforcement or
inspect the CLI's login. Review any deliberately enabled external tools.

## Stored data and deployment files

Private files use owner-only filesystem permissions. They are not encrypted
by ZipperGen. Administrators and processes running as the owner can read them.
Keep the private home directory and its parent directories under the
operator's control. Backups and archived executions need the same care as
active state.

Deployment source is separate from credential storage, but explicitly
including a directory in deployment files includes its contents. Check those
directories for secrets before bundling or sharing them. ZipperGen cannot
identify every filename that might contain a secret.

The execution lock rejects symbolic links, multiply linked files and special
files before changing their permissions or contents. This is additional
protection against a substituted lock file, not protection against an attacker
who controls the account or parent directories.

## Google authorization handoff

The `zg-google-v1...` handoff contains reusable credentials. Its encoding is
not encryption. Its checksum detects copy errors, but is not a signature and
does not authenticate the sender. Keep the entire line private and accept
only a result from your own trusted authorization session.

`zg provider accept` checks the handoff format, required credential fields and
client ID consistency before saving. It does not contact Google to establish
that the token is still valid or that the declared scopes are authentic.
Google remains responsible for enforcing the actual grant on API requests.

## Approvals and recovery

Managed Telegram approvals check the destination chat, configured approver
and token belonging to the pending task. Answering the task and consuming its
token happen in one database transaction. A fresh store uses fresh tokens,
even when task names repeat.

The low-level `TelegramNotifier` API also supports a chat-only policy when
`allowed_user_id` is omitted. Set an explicit approver when using that API in
a group. Treat anyone with access to the private store or local approval CLI
as an operator, not as an untrusted remote user.

Durable recovery does not make arbitrary external effects exactly-once. An
external write can succeed before its local completion record is saved.
Use provider request IDs or idempotent operations where repeating an effect
would be harmful.
