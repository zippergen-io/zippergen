"""Three independent draft approvals using the same persistent participants.

All messages and drafts are fixed examples. No model or email service is used.
"""
from zippergen import Json, Lifeline, Var, human, pure, spawn, workflow

Mailbox, Writer, User = (Lifeline(name) for name in ("Mailbox", "Writer", "User"))
index = Var("index", int, default=0)
completed = Var("completed", Json, default=[])


@pure
def more(index: int) -> bool:
    return index < 3


@pure
def choose_request(index: int) -> str:
    return ("Discuss the paper", "Review the prototype", "Plan the next meeting")[index]


@pure
def advance(index: int) -> int:
    return index + 1


@pure
def draft_reply(request: str) -> str:
    return f"Thanks for your message about '{request}'. When would suit you?"


@human(kind="confirm", outputs=["approved: bool"],
       context="{request}\n\n{draft}", instruction="Accept this example reply?")
def review(request: str, draft: str) -> bool: ...


@pure
def remember(completed: Json, request: str, draft: str) -> Json:
    return [*completed, {"request": request, "draft": draft}]


@workflow
def draft_sessions() -> Json:
    Mailbox: active = more(index)
    while active @ Mailbox:
        Mailbox: request = choose_request(index)
        with spawn(fresh=(
            Mailbox.request, Mailbox.draft, Mailbox.approved,
            Writer.request, Writer.draft,
            User.request, User.draft, User.approved,
        )):
            Mailbox(request) >> Writer(request)
            Writer: draft = draft_reply(request)
            Writer(request, draft) >> User(request, draft)
            User: approved = review(request, draft)
            User(approved, draft) >> Mailbox(approved, draft)
            if approved @ Mailbox:
                Mailbox: completed = remember(completed, request, draft)
        Mailbox: index = advance(index)
        Mailbox: active = more(index)
    return completed @ Mailbox
