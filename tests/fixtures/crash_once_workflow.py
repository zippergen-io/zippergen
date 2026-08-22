"""Subprocess fixture that dies inside one durable external action."""

import json
import os
import sys
from pathlib import Path

from zippergen import Lifeline, Var, effect, workflow
from zippergen.sqlite_runner import run_sqlite


Source = Lifeline("CrashSource")
Worker = Lifeline("CrashWorker")
received = Var("received", int, default=0)
answer = Var("answer", int, default=0)
returned = Var("returned", int, default=0)

marker = Path(sys.argv[2])


@effect
def crash_once(received: int) -> int:
    if not marker.exists():
        marker.write_text("crashed\n")
        os._exit(73)
    return received + 1


@workflow
def crash_round(value: int @ Source) -> int:
    Source(value) >> Worker(received)
    Worker: answer = crash_once(received)
    Worker(answer) >> Source(returned)
    return returned @ Source


result = run_sqlite(
    crash_round,
    [Source, Worker],
    {"CrashSource": {"value": 1}},
    store_path=sys.argv[1],
    timeout=10,
)
print(json.dumps({"result": result}))
