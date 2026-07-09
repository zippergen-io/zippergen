# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false
"""
Hello, ZipperGen — minimal two-lifeline example.

User sends a topic to Writer, Writer runs one LLM action, the result comes back.

Run without an API key:
    python examples/hello.py

To see real results:
    hello.configure("openai:gpt-4o")
    hello.configure("mistral")
    hello.configure("claude")
"""

from zippergen.syntax import Lifeline
from zippergen.actions import llm
from zippergen.builder import workflow

# ---------------------------------------------------------------------------
# Lifelines
# ---------------------------------------------------------------------------

User   = Lifeline("User")
Writer = Lifeline("Writer")

# ---------------------------------------------------------------------------
# LLM actions
# ---------------------------------------------------------------------------

@llm(
    system="Write a concise reply.",
    user="{topic}",
    parse="text",
    outputs=(("draft", str),),
)
def write_reply(topic: str) -> None: ...

# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

@workflow
def hello(topic: str @ User) -> str:
    User(topic) >> Writer(topic)
    Writer: draft = write_reply(topic)
    Writer(draft) >> User(draft)
    return draft @ User

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # No API key needed — runs with the built-in mock backend.
    # Switch to a real LLM: hello.configure("openai:gpt-4o", ui=True)
    hello.configure("mock", ui=True, mock_delay=(0.5, 1.5))
    result = hello(topic="Say hello to ZipperGen")
    print(f"\nResult: {result}")
    try:
        input("ZipperChat is running at http://localhost:8765 . Press Enter to close. ")
    except EOFError:
        pass
