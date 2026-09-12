"""Execution-local FIFO work pools expressed as ordinary effect actions.

Declarations are immutable. Jobs, leases and operation receipts live in the
managed SQLite store, never in the declaration or a Python module global.
Jobs carry CPL context across puts, successful claims and explicit releases.
Pool operations remain local effects; they do not introduce channel messages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import keyword
import math

from zippergen.syntax import EffectAction, Json, Workflow

__all__ = ["Pool", "PoolError", "ClaimExpired", "PoolOperationConflict"]


class PoolError(RuntimeError):
    """A work-pool operation could not be performed."""


class ClaimExpired(PoolError):
    """A claim has expired, been released, or been reassigned."""


class PoolOperationConflict(PoolError):
    """A logical operation was retried with a different request."""


def pool_operations(workflow: Workflow) -> tuple[PoolOperation, ...]:
    """Find used pool declarations through the shared action-site traversal."""
    from zippergen.validation import workflow_actions
    return tuple(
        action.fn for action in workflow_actions(workflow)
        if isinstance(action, EffectAction) and isinstance(action.fn, PoolOperation)
    )


@dataclass(frozen=True)
class PoolOperation:
    """Inspectable implementation descriptor for a built-in effect."""

    pool: str
    operation: str
    lease_seconds: float

    def __call__(self, *args):
        raise PoolError(
            "Pool actions require the SQLite runner. Use the normal workflow "
            "call, 'zg run', or run_sqlite(), rather than the in-memory runner."
        )

    def semantics(self) -> dict[str, object]:
        return {
            "name": self.pool,
            "scope": "execution",
            "order": "fifo-ready",
            "lease_seconds": self.lease_seconds,
            "operation": self.operation,
            "causality": "item-handoff-v1",
        }


@dataclass(frozen=True)
class Pool:
    """Declare a named pool shared by lifelines of one execution.

    ``put`` accepts Json and returns a job ID. ``try_claim`` returns Json:
    None, or a record with pool, job_id, token and payload. ``ack`` and
    ``release`` accept that record and return True on success. Claims have
    a fixed lease (300 seconds by default); they are not renewed implicitly.
    Retried invocations keep their identity, including empty claim results.
    """

    name: str
    lease_seconds: float = 300.0
    put: EffectAction = field(init=False, repr=False, compare=False)
    try_claim: EffectAction = field(init=False, repr=False, compare=False)
    ack: EffectAction = field(init=False, repr=False, compare=False)
    release: EffectAction = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name.isascii()
            or not self.name.isidentifier()
            or keyword.iskeyword(self.name)
        ):
            raise ValueError("A pool name must be an ASCII Python identifier.")
        if (
            isinstance(self.lease_seconds, bool)
            or not isinstance(self.lease_seconds, (int, float))
            or not math.isfinite(self.lease_seconds)
            or self.lease_seconds <= 0
        ):
            raise ValueError("Pool lease_seconds must be finite and positive.")
        object.__setattr__(self, "lease_seconds", float(self.lease_seconds))
        for operation in ("put", "try_claim", "ack", "release"):
            object.__setattr__(self, operation, self._action(operation))

    def _action(self, operation: str) -> EffectAction:
        inputs = (("payload", Json),) if operation == "put" else (
            () if operation == "try_claim" else (("claim", Json),)
        )
        output = str if operation == "put" else (
            Json if operation == "try_claim" else bool
        )
        return EffectAction(
            name=f"{self.name}_{operation}",
            inputs=inputs,
            outputs=(("result", output),),
            fn=PoolOperation(self.name, operation, self.lease_seconds),
        )
