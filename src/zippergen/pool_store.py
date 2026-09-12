"""Atomic pool operations and their durable idempotency receipts.

These transactions are deliberately separate from role advancement. A lost
response repeats the same request against the committed receipt. Optional pool
tables are created lazily in the execution store; ordinary stores are unchanged.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import time

from zippergen.pools import ClaimExpired, PoolError, PoolOperation, PoolOperationConflict
from zippergen.syntax import Json, validate_zvalue


_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS work_pools (
        name TEXT PRIMARY KEY, lease_seconds REAL NOT NULL,
        next_ticket INTEGER NOT NULL DEFAULT 0
    )""",
    """CREATE TABLE IF NOT EXISTS pool_jobs (
        job_id TEXT PRIMARY KEY, pool TEXT NOT NULL, payload TEXT NOT NULL,
        status TEXT NOT NULL, ready_ticket INTEGER NOT NULL,
        owner TEXT, token TEXT, lease_until REAL
    )""",
    """CREATE INDEX IF NOT EXISTS pool_jobs_ready
        ON pool_jobs(pool, status, ready_ticket)""",
    """CREATE INDEX IF NOT EXISTS pool_jobs_expiry
        ON pool_jobs(pool, status, lease_until)""",
    """CREATE TABLE IF NOT EXISTS pool_operations (
        operation_id TEXT PRIMARY KEY, request TEXT NOT NULL, result TEXT NOT NULL
    )""",
)


def pool_operation_id(role: str, steps: int, path: list[int]) -> str:
    """An invocation is a role position/visit, scoped by its execution store.

    Inputs intentionally do not enter the ID: changing them on recovery must
    cause a conflict, rather than create another operation. Loop visits differ
    through steps; paths distinguish parallel action sites at a given position.
    """
    return hashlib.sha256(_json([role, steps, path]).encode()).hexdigest()


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _ticket(conn: sqlite3.Connection, pool: str) -> int:
    conn.execute("UPDATE work_pools SET next_ticket=next_ticket+1 WHERE name=?", (pool,))
    return conn.execute("SELECT next_ticket FROM work_pools WHERE name=?", (pool,)).fetchone()[0]


def _expire(conn: sqlite3.Connection, pool: str, now: float) -> None:
    expired = conn.execute(
        "SELECT job_id FROM pool_jobs WHERE pool=? AND status='claimed' "
        "AND lease_until<=? ORDER BY lease_until, ready_ticket", (pool, now),
    ).fetchall()
    for (job_id,) in expired:
        conn.execute(
            "UPDATE pool_jobs SET status='ready', ready_ticket=?, owner=NULL, "
            "token=NULL, lease_until=NULL WHERE job_id=?",
            (_ticket(conn, pool), job_id),
        )


def _claim_record(value: object, pool: str) -> dict:
    if not isinstance(value, dict) or value.get("pool") != pool:
        raise PoolError("Expected a claim from this pool.")
    if not all(isinstance(value.get(k), str) and value[k] for k in ("job_id", "token")):
        raise PoolError("A claim needs a job_id and token.")
    return value


def _perform(conn, operation: PoolOperation, owner: str, inputs: dict, now: float):
    pool = operation.pool
    if operation.operation == "put":
        job_id = secrets.token_hex(16)
        conn.execute(
            "INSERT INTO pool_jobs(job_id,pool,payload,status,ready_ticket) "
            "VALUES(?,?,?,'ready',?)",
            (job_id, pool, _json(inputs["payload"]), _ticket(conn, pool)),
        )
        return job_id

    if operation.operation == "try_claim":
        job = conn.execute(
            "SELECT job_id,payload FROM pool_jobs WHERE pool=? AND status='ready' "
            "ORDER BY ready_ticket LIMIT 1", (pool,),
        ).fetchone()
        if job is None:
            return None
        token = secrets.token_hex(24)
        conn.execute(
            "UPDATE pool_jobs SET status='claimed',owner=?,token=?,lease_until=? "
            "WHERE job_id=?", (owner, token, now + operation.lease_seconds, job[0]),
        )
        return {"pool": pool, "job_id": job[0], "token": token, "payload": json.loads(job[1])}

    claim = _claim_record(inputs["claim"], pool)
    job = conn.execute(
        "SELECT status,owner,token FROM pool_jobs WHERE pool=? AND job_id=?",
        (pool, claim["job_id"]),
    ).fetchone()
    if job is None or job[1] != owner or job[2] != claim["token"]:
        raise ClaimExpired("Claim is expired, released, or belongs to another worker.")
    if operation.operation == "ack" and job[0] == "completed":
        return True
    if job[0] != "claimed":
        raise ClaimExpired("Claim is no longer active.")
    if operation.operation == "ack":
        conn.execute(
            "UPDATE pool_jobs SET status='completed',lease_until=NULL WHERE job_id=?",
            (claim["job_id"],),
        )
    elif operation.operation == "release":
        conn.execute(
            "UPDATE pool_jobs SET status='ready',ready_ticket=?,owner=NULL, "
            "token=NULL,lease_until=NULL WHERE job_id=?",
            (_ticket(conn, pool), claim["job_id"]),
        )
    else:
        raise PoolError(f"Unknown pool operation: {operation.operation}.")
    return True


def execute_pool_operation(
    conn: sqlite3.Connection, operation: PoolOperation, *,
    operation_id: str, owner: str, inputs: dict,
):
    """Commit a request and its result together, or return its saved result.

    A replayed claim is a historical result, not a lease renewal. Its token
    cannot acknowledge a job after expiry or reassignment. Empty results are
    receipts too: the next poll must be a new logical invocation.
    """
    if conn.in_transaction:
        raise PoolError("A pool operation must run outside the role transaction.")
    expected = {"payload"} if operation.operation == "put" else (
        set() if operation.operation == "try_claim" else {"claim"}
    )
    if operation.operation not in {"put", "try_claim", "ack", "release"} or set(inputs) != expected:
        raise PoolError("Invalid pool operation inputs.")
    validate_zvalue(inputs, Json, context="Pool operation inputs")
    request = _json({"pool": operation.semantics(), "owner": owner, "inputs": inputs})
    conn.execute("BEGIN IMMEDIATE")
    try:
        for statement in _SCHEMA:
            conn.execute(statement)
        previous = conn.execute(
            "SELECT request,result FROM pool_operations WHERE operation_id=?", (operation_id,),
        ).fetchone()
        if previous is not None:
            if previous[0] != request:
                raise PoolOperationConflict("This pool invocation was already executed with a different request.")
            result = json.loads(previous[1])
        else:
            conn.execute(
                "INSERT INTO work_pools(name,lease_seconds) VALUES(?,?) ON CONFLICT(name) DO NOTHING",
                (operation.pool, operation.lease_seconds),
            )
            lease = conn.execute(
                "SELECT lease_seconds FROM work_pools WHERE name=?", (operation.pool,),
            ).fetchone()[0]
            if lease != operation.lease_seconds:
                raise PoolError("Pool lease configuration differs from its saved declaration.")
            now = time.time()
            _expire(conn, operation.pool, now)
            result = _perform(conn, operation, owner, inputs, now)
            conn.execute(
                "INSERT INTO pool_operations(operation_id,request,result) VALUES(?,?,?)",
                (operation_id, request, _json(result)),
            )
        conn.execute("COMMIT")
        return result
    except BaseException:
        conn.execute("ROLLBACK")
        raise
