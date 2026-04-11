# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false
"""
Debate — live demo example.

Three lifelines debate a topic until the Judge decides to stop:
  Pro     — argues in favour of the motion
  Con     — argues against the motion
  Judge   — evaluates each exchange and decides when to call a verdict

The loop is owned by Judge: each round Pro and Con exchange rebuttals,
then Judge assesses whether the debate is settled. This demonstrates
ZipperGen's deadlock-free coordination guarantee across a non-trivial
three-party loop with cross-lifeline message exchange.

Usage:
  python debate.py "AI should replace human judges in courtrooms"
  python debate.py          # prompts for a topic at runtime
"""

import sys
from zippergen.syntax import Lifeline, Var
from zippergen.actions import llm, pure
from zippergen.builder import workflow

MAX_ROUNDS = 4

# ---------------------------------------------------------------------------
# Lifelines
# ---------------------------------------------------------------------------

Host  = Lifeline("Host")
Pro   = Lifeline("Pro")
Con   = Lifeline("Con")
Judge = Lifeline("Judge")

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

topic          = Var("topic",          str)
argument       = Var("argument",       str)   # own current argument (Pro / Con local)
other_argument = Var("other_argument", str)   # opponent's latest argument
pro_arg        = Var("pro_arg",        str)   # Pro's argument as seen by Judge
con_arg        = Var("con_arg",        str)   # Con's argument as seen by Judge
verdict        = Var("verdict",        str,   default="")
go_on          = Var("go_on",          bool,  default=True)
rounds         = Var("rounds",         int,   default=0)

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@llm(
    system=(
        "You are a skilled debater. Your role is to argue strongly IN FAVOUR of "
        "the given topic. Be concise (3-5 sentences), persuasive, and use evidence "
        "or logic. Respond with JSON: {\"argument\": \"...\"}."
    ),
    user="Topic: {topic}\n\nMake your opening argument.",
    parse="json",
    outputs=(("argument", str),),
)
def open_for(topic: str) -> None: ...


@llm(
    system=(
        "You are a skilled debater. Your role is to argue strongly AGAINST "
        "the given topic. Be concise (3-5 sentences), persuasive, and use evidence "
        "or logic. Respond with JSON: {\"argument\": \"...\"}."
    ),
    user="Topic: {topic}\n\nMake your opening argument.",
    parse="json",
    outputs=(("argument", str),),
)
def open_against(topic: str) -> None: ...


@llm(
    system=(
        "You are a skilled debater arguing IN FAVOUR of the topic. "
        "You have heard your opponent's latest argument. Rebut it directly, "
        "then reinforce your own position. Be concise (3-5 sentences). "
        "Respond with JSON: {\"argument\": \"...\"}."
    ),
    user=(
        "Topic: {topic}\n\n"
        "Your previous argument: {argument}\n\n"
        "Opponent said: {other_argument}\n\n"
        "Rebut and reinforce."
    ),
    parse="json",
    outputs=(("argument", str),),
)
def rebut_for(topic: str, argument: str, other_argument: str) -> None: ...


@llm(
    system=(
        "You are a skilled debater arguing AGAINST the topic. "
        "You have heard your opponent's latest argument. Rebut it directly, "
        "then reinforce your own position. Be concise (3-5 sentences). "
        "Respond with JSON: {\"argument\": \"...\"}."
    ),
    user=(
        "Topic: {topic}\n\n"
        "Your previous argument: {argument}\n\n"
        "Opponent said: {other_argument}\n\n"
        "Rebut and reinforce."
    ),
    parse="json",
    outputs=(("argument", str),),
)
def rebut_against(topic: str, argument: str, other_argument: str) -> None: ...


@llm(
    system=(
        "You are an impartial debate judge. After reading the latest arguments "
        "from both sides, decide whether the debate should continue for another "
        "round. Continue if the arguments are still evolving; stop if one side "
        "has clearly prevailed or the exchange is becoming repetitive. "
        "Respond with JSON: {\"go_on\": true/false}."
    ),
    user=(
        "Topic: {topic}\n\n"
        "Pro: {pro_arg}\n\n"
        "Con: {con_arg}\n\n"
        "Should the debate continue?"
    ),
    parse="json",
    outputs=(("go_on", bool),),
)
def assess(topic: str, pro_arg: str, con_arg: str) -> None: ...


@llm(
    system=(
        "You are an impartial debate judge. The debate has concluded. "
        "Weigh both sides carefully and deliver your final verdict: "
        "which position was better argued, and why? Be decisive — name a winner. "
        "Respond with JSON: {\"verdict\": \"...\"}."
    ),
    user=(
        "Topic: {topic}\n\n"
        "Final arguments:\n\nPro: {pro_arg}\n\nCon: {con_arg}\n\n"
        "Deliver your verdict."
    ),
    parse="json",
    outputs=(("verdict", str),),
)
def conclude(topic: str, pro_arg: str, con_arg: str) -> None: ...


@pure
def inc_rounds(r: int) -> int:
    return r + 1


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

@workflow
def debate(topic: str @ Host) -> str:
    # Distribute topic to all participants
    Host(topic) >> Pro(topic)
    Host(topic) >> Con(topic)
    Host(topic) >> Judge(topic)

    # Opening arguments
    Pro:  argument = open_for(topic)
    Con:  argument = open_against(topic)

    # Debaters see each other's opening
    Pro(argument) >> Con(other_argument)
    Con(argument) >> Pro(other_argument)

    # Judge receives both openings
    Pro(argument) >> Judge(pro_arg)
    Con(argument) >> Judge(con_arg)

    # Initial evaluation — Judge decides whether to open the rebuttal loop
    Judge: go_on = assess(topic, pro_arg, con_arg)

    # Rebuttal loop — Judge owns the continuation guard
    while (go_on and rounds < MAX_ROUNDS) @ Judge:
        Pro:  argument = rebut_for(topic, argument, other_argument)
        Con:  argument = rebut_against(topic, argument, other_argument)

        # Exchange rebuttals
        Pro(argument) >> Con(other_argument)
        Con(argument) >> Pro(other_argument)

        # Judge receives the new round
        Pro(argument) >> Judge(pro_arg)
        Con(argument) >> Judge(con_arg)

        with Judge:
            go_on = assess(topic, pro_arg, con_arg)
            rounds = inc_rounds(rounds)

    # Judge delivers final verdict after the loop
    Judge: verdict = conclude(topic, pro_arg, con_arg)

    # Deliver verdict to Host
    Judge(verdict) >> Host(verdict)
    return verdict @ Host


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        topic_text = " ".join(sys.argv[1:])
    else:
        topic_text = input("Debate topic: ").strip()
        if not topic_text:
            topic_text = "AI will make human judges obsolete"

    USE_UI = True

    debate.configure(
        llms={"Pro": "openai", "Con": "openai", "Judge": "openai"},
        # llms="mock",
        ui=USE_UI,
        timeout=600,
    )
    result = debate(topic=topic_text)
    print(f"\nVerdict → {result}")
    if USE_UI:
        input("ZipperChat is running at http://localhost:8765 . Press Enter to close. ")
