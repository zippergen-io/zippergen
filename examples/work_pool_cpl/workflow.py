"""An approval guard receives its evidence through the claimed job."""
import time

from zippergen import At, Here, Json, Lifeline, Pool, effect, pure, workflow

Producer = Lifeline("Producer")
Worker = Lifeline("Worker")
jobs = Pool("jobs")

approved_for_job = (
    (At[Producer].approved == True)
    & (At[Producer].request == Here.request)
    & (At[Producer].job_id == Here.job_id)
)


@pure
def request_name() -> str:
    return "report-17"


@pure
def review(allow: bool) -> bool:
    return allow


@pure
def job_payload(request: str) -> Json:
    return {"request": request}


@pure
def request_of(claim: Json) -> str:
    return claim["payload"]["request"]


@pure
def id_of(claim: Json) -> str:
    return claim["job_id"]


@effect
def wait_for_work() -> bool:
    time.sleep(0.01)
    return True


@pure
def process(request: str) -> str:
    return f"processed {request}"


@pure
def reject(request: str) -> str:
    return f"rejected {request}"


@workflow
def approved_work_pool(allow: bool @ Producer) -> str:
    Producer: request = request_name()
    Producer: approved = review(allow)
    Producer: payload = job_payload(request)
    Producer: job_id = jobs.put(payload)

    Worker: claim = jobs.try_claim()
    while (claim is None) @ Worker:
        Worker: waited = wait_for_work()
        Worker: claim = jobs.try_claim()
    Worker: request = request_of(claim)
    Worker: job_id = id_of(claim)
    if approved_for_job @ Worker:
        Worker: result = process(request)
    else:
        Worker: result = reject(request)
    Worker: completed = jobs.ack(claim)
    return result @ Worker
