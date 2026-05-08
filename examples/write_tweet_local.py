# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false
"""
Self-contained write-tweet example using a local OpenAI-compatible model server.

Start a local server first, for example on a GPU machine:

    VLLM_USE_DEEP_GEMM=0 vllm serve Qwen/Qwen2.5-7B-Instruct \
      --host 127.0.0.1 --port 8000

If the server is remote, forward the port from your local machine:

    ssh -L 8000:127.0.0.1:8000 lmf-gpu

Then run:

    python examples/write_tweet_local.py
"""

from zippergen import Lifeline, Var, llm, workflow
from zippergen.backends import make_openai_backend


User   = Lifeline("User")
Writer = Lifeline("Writer")
Editor = Lifeline("Editor")

topic    = Var("topic",    str)
tweet    = Var("tweet",    str)
approved = Var("approved", bool)


@llm(
    system="Write one concise English tweet about the topic. Do not use hashtags or emojis.",
    user="{topic}",
    parse="text",
    outputs=(("tweet", str),),
)
def draft(topic: str) -> None: ...


@llm(
    system=(
        "Is this tweet in English, concise, and under 180 characters? "
        "Reply true or false."
    ),
    user="{tweet}",
    parse="bool",
    outputs=(("approved", bool),),
)
def approve(tweet: str) -> None: ...


@llm(
    system=(
        "Rewrite this as one concise English tweet under 180 characters. "
        "Do not use hashtags or emojis."
    ),
    user="{tweet}",
    parse="text",
    outputs=(("tweet", str),),
)
def revise(tweet: str) -> None: ...


@workflow
def write_tweet_local(topic: str @ User) -> str:
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


if __name__ == "__main__":
    backend = make_openai_backend(
        api_key="EMPTY",
        model="Qwen/Qwen2.5-7B-Instruct",
        base_url="http://127.0.0.1:8000/v1",
        max_tokens=256,
        timeout=120,
    )
    write_tweet_local.configure(backend=backend, ui=True, timeout=300)
    result = write_tweet_local(topic="a git commit message that tells the truth")
    print(f"\nResult: {result}")
    input("ZipperChat is running at http://localhost:8765 . Press Enter to close. ")
