# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportUndefinedVariable=false, reportReturnType=false
"""Two assistants work a task; a human decides whether they continue.

An implementer proposes a change and an adversarial reviewer attacks it. Each
round the human sees one line from each and answers: ship it, another round, or
abandon. Both assistants wait on that answer -- they never evaluate the loop
themselves, which is what ``zg show --agent Reviewer`` makes visible.

Both assistants may be the same CLI. Every ``@assistant`` call is a fresh
non-persistent session, so an Implementer and a Reviewer routed to the same
executable share no context and the review stays independent. Their
capabilities differ where it matters: the implementer may write, the reviewer
may only read.

Setup, once per project
-----------------------

    zg assistant check                      # is claude or codex usable
    zg assistant configure impl claude
    zg assistant configure rev  claude      # same CLI, separate session
    zg assistant assign Implementer impl
    zg assistant assign Reviewer     rev
    zg validate

Running it
----------

    zg run --yes --durable --timeout 0

The declared configuration below supplies ``max_rounds`` and ``detail``, and
remembers whatever you answer, so a later run needs no flags at all. Override
one for a single run, and that answer is remembered too:

    zg run --input 'task=Fix the flaky projection test.' \
           --input detail=brief --yes --durable --timeout 0

``zg config`` shows the current answers; ``--durable --timeout 0`` is what lets
a round take as long as an assistant needs and survive a crash.

``detail`` decides how much reaches the person:

    full    the whole proposal and the whole review
    brief   one line from each, and where to read the rest
    auto    nobody is asked; the reviewer's own verdict ends or continues it

Deploying it
------------

    zg deploy --yes

Deployed, the human answers over Telegram: a choice for the verdict, and a
free-text reply when another round is wanted. Deploy it on the machine that
holds the repository being worked on -- an ``@assistant`` action shells out to
a local CLI, so the workflow and the code must be on the same host.

``workspace`` below names the repository the assistants read and edit. Give it
an absolute path: a relative one is resolved against the root the workflow runs
from, and a deployment runs from an immutable bundle rather than this
directory, so ``../target`` would work for ``zg run`` and fail once deployed.
``zg deploy`` checks this before it applies anything.
"""

from zippergen import Lifeline, assistant, human, pure, workflow
from zippergen.deployment import DeploymentField, DeploymentSpec


# The repository these assistants read and edit. Set it before running: an
# absolute path, because a relative one is resolved against the root the
# workflow runs from, and a deployment runs from an immutable bundle rather
# than from your project. Left as a placeholder on purpose -- it does not
# exist, so `zg deploy` refuses rather than pointing an assistant with write
# access at the wrong directory.
WORKSPACE = "/absolute/path/to/the/repository/to/work/on"


# Values that belong to this deployment rather than to one invocation. Declared
# once here, answered once at deploy time, and stored in the deployment profile
# -- so neither a person nor a command line has to carry them around.
zippergen_deployment = DeploymentSpec(
    description="Two assistants work a task; a human validates every turn.",
    fields=(
        DeploymentField(
            "task",
            "What should the assistants work on?",
            target="input",
            required=True,
        ),
        DeploymentField(
            "max_rounds",
            "Maximum implement/review rounds before giving up",
            target="input",
            default=4,
            required=True,
        ),
        DeploymentField(
            "detail",
            "How much the human is told each turn",
            target="input",
            default="brief",
            required=True,
            choices=("brief", "full", "auto"),
        ),
    ),
    files=("workflow.py", "prompts/implement.md", "prompts/review.md"),
)


Human = Lifeline("Human")
Implementer = Lifeline("Implementer")
Reviewer = Lifeline("Reviewer")


# --- the two assistants -------------------------------------------------
# The implementer writes; the reviewer only reads. That asymmetry is the
# point: a reviewer that can edit the code stops being a second opinion.

@assistant(
    instructions_file="examples/prompts/pair_programming/implement.md",
    access="write",
    external_tools="none",
    shell="enabled",
    workspace=WORKSPACE,
)
def implement(task: str, guidance: str) -> str: ...


@assistant(
    instructions_file="examples/prompts/pair_programming/review.md",
    access="read-only",
    external_tools="none",
    shell="enabled",
    workspace=WORKSPACE,
)
def review(task: str, proposal: str) -> str: ...


# --- the human turn -----------------------------------------------------
# One select decides the turn. Free-text guidance is asked for separately,
# and only when it is actually needed, so an ordinary approval stays one tap.

# The labels name the outcome, not the feeling. "Accept" was read as
# "accept the reviewer's findings" -- the opposite of what it does, which is
# to take the proposal and finish without another round.
SHIP = "Ship it"
ANOTHER_ROUND = "Another round"
ABANDON = "Abandon"


@human(
    kind="select",
    context="{briefing}",
    instruction=(
        "Round {rounds} of {max_rounds}. "
        "'Ship it' keeps this proposal and ends the run now. "
        "'Another round' sends the review back to the implementer, then asks "
        "you whether to add anything. "
        "'Abandon' stops and keeps nothing. "
        "'Ship it' and 'Abandon' end the run without asking anything further."
    ),
    prefill=f"{SHIP}\n{ANOTHER_ROUND}\n{ABANDON}",
    outputs=["verdict: str"],
)
def judge_turn(briefing: str, rounds: int, max_rounds: int): ...


@human(
    kind="input",
    instruction=(
        "Anything to add for the implementer? It already receives the "
        "reviewer's full findings. Add only what you want emphasised, "
        "overruled, or done differently. Leave empty to pass the review "
        "through unchanged."
    ),
    outputs=["note: str"],
)
def ask_note(): ...


# --- what the human is told, and whether they are asked --------------------
# Two separate questions, kept separate. The assistants write for each other;
# `detail` decides how much of that reaches a person. "auto" answers the turn
# from the reviewer's own verdict and never interrupts anyone.
#
# The first three lines of a review are a declared contract (see
# prompts/review.md), not prose to be guessed at. A review that does not meet
# it says so plainly rather than being silently misread.

VERDICT_LINE = 0
FINDINGS_LINE = 1
HEADLINE_LINE = 2


@pure
def review_header(critique: str) -> str:
    """Return 'VERDICT|FINDINGS|HEADLINE', or an explicit complaint."""
    lines = [line.strip() for line in critique.strip().splitlines()]
    if len(lines) < 3:
        return "REVISE|?|The reviewer did not use the required three-line header."
    verdict = lines[VERDICT_LINE].casefold()
    if verdict not in {"approve", "revise"}:
        return "REVISE|?|The reviewer's first line was not APPROVE or REVISE."
    findings = lines[FINDINGS_LINE]
    prefix = "findings:"
    count = (
        findings[len(prefix):].strip()
        if findings.casefold().startswith(prefix)
        else "?"
    )
    return f"{lines[VERDICT_LINE].upper()}|{count}|{lines[HEADLINE_LINE]}"


@pure
def briefing(proposal: str, critique: str, header: str, detail: str) -> str:
    """Render the turn for a person, at the requested depth."""
    verdict, count, headline = header.split("|", 2)
    if detail == "full":
        return f"Proposal:\n\n{proposal}\n\nReview:\n\n{critique}"
    return (
        f"Implementer: {proposal_headline.fn(proposal)}\n\n"
        f"Reviewer: {verdict} ({count} finding(s))\n"
        f"{headline}\n\n"
        "Full text: zg run trace --tail 20 --json"
    )


@pure
def asks_the_human(detail: str) -> bool:
    return detail != "auto"


@pure
def verdict_from_review(header: str) -> str:
    return SHIP if header.split("|", 1)[0] == "APPROVE" else ANOTHER_ROUND


@pure
def no_guidance() -> str:
    return "No additional guidance; follow the task as written."


@pure
def guidance_from_review(critique: str, note: str) -> str:
    """What the implementer is told: always the review, plus any human note.

    Interactive and automatic turns build this the same way. Sending only the
    human's words is how an implementer came to be told "The revision is ok."
    with no idea what the reviewer had said.
    """
    briefing = f"Address this review:\n\n{critique}"
    if not note.strip():
        return briefing
    return f"{briefing}\n\nThe human adds:\n\n{note.strip()}"


@pure
def no_note() -> str:
    return ""


# `keep_going` answers one question -- will there be another round? -- and both
# the loop and the note depend on it. Asking it once is what keeps the workflow
# from requesting guidance it is about to discard: a "Revise" on the final round
# ends the run, so there is nobody left to read the note.


@pure
def proposal_headline(proposal: str) -> str:
    """The implementer's own one-line summary (see prompts/implement.md)."""
    for line in proposal.strip().splitlines():
        if line.strip():
            return line.strip()
    return "The implementer returned nothing."


@pure
def first_round() -> int:
    return 1


@pure
def next_round(rounds: int) -> int:
    return rounds + 1


@pure
def keep_going(verdict: str, rounds: int, max_rounds: int) -> bool:
    return (
        verdict.strip().casefold() == ANOTHER_ROUND.casefold()
        and rounds < max_rounds
    )


@pure
def outcome(
    verdict: str,
    task: str,
    proposal: str,
    critique: str,
    rounds: int,
) -> str:
    """Say what happened, what is in the repository, and what is not settled.

    The result is the last thing a person reads, often hours later. Returning
    the raw proposal made them reconstruct the run from it; this says which
    ending occurred, what the reviewer still held against the work, and where
    the full text is.
    """

    decision = verdict.strip().casefold()
    lines = [f"Task: {task.strip()}"]
    if decision == SHIP.casefold():
        lines.append(
            f"Shipped after {rounds} round(s). The change below is in the "
            "repository."
        )
    elif decision == ABANDON.casefold():
        lines.append(
            f"Abandoned after {rounds} round(s). Whatever the implementer "
            "last wrote is still in the working tree; revert it if you do "
            "not want it."
        )
    else:
        lines.append(
            f"Stopped at the {rounds}-round limit, still unresolved. The last "
            "proposal is in the working tree and the reviewer had not "
            "approved it."
        )

    verdict_line, _, headline = review_header.fn(critique).split("|", 2)
    lines.append("")
    lines.append(f"Reviewer's last word: {verdict_line} -- {headline}")
    lines.append("")
    lines.append("What the implementer did last:")
    lines.append(proposal_headline.fn(proposal))
    lines.append("")
    lines.append(
        "Full proposals and reviews: zg run trace --tail 40 --json"
    )
    lines.append("Uncommitted changes: git diff in the workspace")
    return "\n".join(lines)


@workflow
def pair_programming(
    task: str @ Human,
    max_rounds: int @ Human,
    detail: str @ Human,
) -> str:
    Human(task) >> Implementer(task)
    Human(task) >> Reviewer(task)

    with Human:
        guidance = no_guidance()
        rounds = first_round()
        ask = asks_the_human(detail)
    Human(guidance) >> Implementer(guidance)

    Implementer: proposal = implement(task, guidance)
    Implementer(proposal) >> Reviewer(proposal)
    Reviewer: critique = review(task, proposal)

    Implementer(proposal) >> Human(proposal)
    Reviewer(critique) >> Human(critique)
    with Human:
        header = review_header(critique)
        brief = briefing(proposal, critique, header, detail)
    if ask @ Human:
        with Human:
            verdict = judge_turn(brief, rounds, max_rounds)
            again = keep_going(verdict, rounds, max_rounds)
        if again @ Human:
            Human: note = ask_note()
        else:
            Human: note = no_note()
    else:
        with Human:
            verdict = verdict_from_review(header)
            again = keep_going(verdict, rounds, max_rounds)
            note = no_note()
    Human: guidance = guidance_from_review(critique, note)

    while again @ Human:
        Human: rounds = next_round(rounds)
        Human(guidance) >> Implementer(guidance)

        Implementer: proposal = implement(task, guidance)
        Implementer(proposal) >> Reviewer(proposal)
        Reviewer: critique = review(task, proposal)

        Implementer(proposal) >> Human(proposal)
        Reviewer(critique) >> Human(critique)
        with Human:
            header = review_header(critique)
            brief = briefing(proposal, critique, header, detail)
        if ask @ Human:
            with Human:
                verdict = judge_turn(brief, rounds, max_rounds)
                again = keep_going(verdict, rounds, max_rounds)
            if again @ Human:
                Human: note = ask_note()
            else:
                Human: note = no_note()
        else:
            with Human:
                verdict = verdict_from_review(header)
                again = keep_going(verdict, rounds, max_rounds)
                note = no_note()
        Human: guidance = guidance_from_review(critique, note)

    Human: result = outcome(verdict, task, proposal, critique, rounds)
    return result @ Human
