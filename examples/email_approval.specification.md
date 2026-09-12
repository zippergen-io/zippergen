# Email approval tutorial

One workflow watches one directory of plain text requests. Each request has a
unique .txt filename. Inputs remain unchanged until the workflow marks them
handled. Writer drafts a reply. Mailbox asks a person whether to send it.
Approval prints a simulated send and increments the approved count. Rejection
does not send. Both outcomes mark the input .done and count toward an optional
max_messages budget. Without that budget, wait for more requests indefinitely.

The filename, content, and processed count belong to durable workflow state.
Reading must not remove a request. Mark it done only after handling it, and
recognize a repeated completion if the process stopped after the rename but
before recording its result. An external sending service would also need a
stable idempotency key. The tutorial only prints a send notification, which can
repeat if that print was not committed before a crash.

## Change on 12 September 2026

Replaced destructive reads with a durable item record and explicit completion.
Moved the optional budget's progress from a Python global to a workflow
variable. This changes the protocol, so saved executions of the old tutorial
must finish with the old version or be reset before using this version.
