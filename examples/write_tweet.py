# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false
"""
Write Tweet — Hello World example.

Three agents collaborate: Writer drafts a tweet, Editor decides whether it
is good enough, and ZipperGen handles the coordination automatically.

Run without an API key:
    python examples/write_tweet.py

The built-in mock backend produces placeholder output. To see real results:
    write_tweet.configure(llms="openai")
    write_tweet.configure(llms="mistral")
    write_tweet.configure(llms="claude")
"""

from zippergen.syntax import Lifeline, Var
from zippergen.actions import llm
from zippergen.builder import workflow

# ---------------------------------------------------------------------------
# Lifelines
# ---------------------------------------------------------------------------

User   = Lifeline("User")
Writer = Lifeline("Writer")
Editor = Lifeline("Editor")

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

topic    = Var("topic",    str)
tweet    = Var("tweet",    str)
approved = Var("approved", bool)

# ---------------------------------------------------------------------------
# LLM actions
# ---------------------------------------------------------------------------

@llm(
    system="Write a tweet about the topic.",
    user="{topic}",
    parse="text",
    outputs=(("tweet", str),),
)
def draft(topic: str) -> None: ...


@llm(
    system=(
        "Is this tweet engaging, original, and under 180 characters? "
        "Reply true or false."
    ),
    user="{tweet}",
    parse="bool",
    outputs=(("approved", bool),),
)
def approve(tweet: str) -> None: ...


@llm(
    system="Improve this tweet: shorter and punchier.",
    user="{tweet}",
    parse="text",
    outputs=(("tweet", str),),
)
def revise(tweet: str) -> None: ...

# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

@workflow
def write_tweet(topic: str @ User) -> str:
    User(topic) >> Writer(topic)
    Writer: tweet = draft(topic)
    Writer(tweet) >> Editor(tweet)
    Editor: approved = approve(tweet)
    if approved @ Editor:
        Editor(tweet) >> User(tweet)
    else:
        Editor(tweet) >> Writer(tweet)
        Writer: tweet = revise(tweet)
        Writer(tweet) >> User(tweet)
    return tweet @ User

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # No API key needed — runs with the built-in mock backend.
    # Switch to a real LLM: write_tweet.configure(llms="openai")
    write_tweet.configure(llms="mock", ui=True, mock_delay=(0.5, 1.5))
    result = write_tweet(topic="a git commit message that tells the truth")
    print(f"\nResult: {result}")
    input("ZipperChat is running at http://localhost:8765 . Press Enter to close. ")
