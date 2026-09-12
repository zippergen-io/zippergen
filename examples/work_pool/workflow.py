"""Two consumers of an execution-local FIFO pool; no services or keys needed."""
from zippergen import Json, Lifeline, Pool, Var, branch, parallel, pure, workflow

Producer = Lifeline("Producer")
WorkerA = Lifeline("WorkerA")
WorkerB = Lifeline("WorkerB")
jobs = Pool("jobs")
first_result = Var("first_result", str)
second_result = Var("second_result", str)


@pure
def process(claim: Json) -> str:
    return f"processed job {claim['payload']['number']}"


@pure
def no_job() -> str:
    return "no job available"


@pure
def summarize(first: str, second: str) -> str:
    return "; ".join(sorted((first, second)))


@workflow
def work_pool() -> str:
    Producer: first = jobs.put({"number": 1})
    Producer: second = jobs.put({"number": 2})
    Producer(second) >> WorkerA(second)
    Producer(second) >> WorkerB(second)
    with parallel:
        with branch:
            WorkerA: claim = jobs.try_claim()
            if (claim is not None) @ WorkerA:
                WorkerA: result = process(claim)
                WorkerA: completed = jobs.ack(claim)
            else:
                WorkerA: result = no_job()
        with branch:
            WorkerB: claim = jobs.try_claim()
            if (claim is not None) @ WorkerB:
                WorkerB: result = process(claim)
                WorkerB: completed = jobs.ack(claim)
            else:
                WorkerB: result = no_job()
    WorkerA(result) >> Producer(first_result)
    WorkerB(result) >> Producer(second_result)
    Producer: summary = summarize(first_result, second_result)
    return summary @ Producer
