"""One retry policy for one LLM attempt, and the errors it decides on.

A model call is not finished when the HTTP response arrives. It is finished
when the response has been parsed, coerced and checked against the action's
declared outputs. A model that answers ``maybe`` to a boolean question has
failed just as surely as one that refused the connection, and it fails in a
way that asking again may well fix.

So the retry loop lives here, around the whole attempt, and the HTTP backend
has none of its own. What the backend contributes is classification: it knows
that 429 is worth waiting for and 401 never will be, and it says so by raising
one of the errors below.

Three kinds of failure, because they need three different answers:

    transient          the call did not happen; try again
    invalid response   the call happened and the answer is unusable; try again
    permanent          the call will fail the same way forever; stop

Anything that is not one of these is a defect -- in ZipperGen, in a workflow's
own code, or in the store -- and is left to propagate. A retry loop that
swallows ``Exception`` turns a typo into an infinite wait.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from typing import Callable

from zippergen.syntax import validate_zvalue
from zippergen.value_codec import loads_value
from zippergen.errors import WorkflowCancelled


__all__ = [
    "FOREVER",
    "LLMError",
    "LLMInvalidResponseError",
    "LLMPermanentError",
    "LLMTransientError",
    "RetryCancelled",
    "attempt_llm_action",
    "checked_llm_outputs",
    "retry_reporter",
    "retry_delays",
]


#: The literal accepted by ``retries=`` for an unbounded budget.
FOREVER = "forever"

#: Exponential backoff, capped. A model that is rate-limiting or restarting
#: recovers on the order of seconds, and waiting longer than half a minute
#: between attempts only delays the recovery it is waiting for.
_FIRST_DELAY = 2.0
_MAX_DELAY = 30.0


class LLMError(RuntimeError):
    """A recognised failure of one LLM call."""


class LLMTransientError(LLMError):
    """The call did not complete, and the same call may succeed later.

    Network failures, timeouts, 429 and the transient 5xx responses. Carries
    ``retry_after`` when the provider said how long to wait, which is always
    better information than a backoff schedule can guess.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMInvalidResponseError(LLMError):
    """The call completed and the answer cannot be used.

    Unparseable JSON, a missing output key, a value of the wrong type. The
    request was well formed, so asking again is reasonable: the next sample
    may well be valid.
    """


class LLMPermanentError(LLMError):
    """The call will fail identically however often it is repeated.

    A rejected key, a model that does not exist, a malformed request. Retrying
    wastes time and hides the cause.
    """


class RetryCancelled(WorkflowCancelled):
    """The run or deployment was stopped while waiting to try again.

    Subclasses ``WorkflowCancelled``, which is how the runners tell a stop
    apart from a fault, so a cancelled wait is not recorded as a failure and
    cannot be chosen as the root cause ahead of the error that set the stop
    event.
    """

    def __init__(
        self, detail: str = "stopped while waiting to retry an LLM call"
    ) -> None:
        super().__init__(f"Workflow cancelled: {detail}")


def retry_reporter(trace, action) -> Callable[[str], None]:
    """Report retries through the normal trace, or plainly on stderr.

    Keeping this beside the retry loop gives ordinary ``@llm`` calls and the
    LLM calls made internally by ``@planner`` exactly the same observable
    contract.  Trace consumers expect ``type`` on every event.
    """

    def report(message: str) -> None:
        if trace is not None:
            trace({
                "type": "llm_retry",
                "action": action.name,
                "action_kind": "llm",
                "detail": message,
            })
        else:
            print(f"[zippergen] {message}", file=sys.stderr)

    return report


def retry_delays(retry_after: float | None, attempt: int) -> float:
    """How long to wait before attempt ``attempt`` (0 for the first retry).

    A provider's own ``Retry-After`` wins outright and is *not* capped: it is
    an instruction, not a guess, and ignoring a request for two minutes only
    earns another rate limit. The cap applies to the schedule ZipperGen
    invents for itself -- 2, 4, 8 seconds and so on -- so an unbounded budget
    does not drift into waiting minutes for no reason.
    """

    if retry_after is not None and retry_after >= 0:
        return float(retry_after)
    return min(_FIRST_DELAY * (2.0**attempt), _MAX_DELAY)


def _sleep_or_cancel(delay: float, stop: threading.Event | None) -> None:
    """Wait, unless the run is stopping.

    ``Event.wait`` returns as soon as the event is set, so a stop during a
    thirty-second backoff is noticed immediately rather than after it.
    """

    if stop is None:
        threading.Event().wait(delay)
        return
    if stop.wait(delay):
        raise RetryCancelled("Stopped while waiting to retry an LLM call.")


@dataclass(frozen=True)
class _Budget:
    """How many more attempts remain, and whether that number is finite."""

    limit: int | str

    @property
    def unbounded(self) -> bool:
        return self.limit == FOREVER

    def exhausted(self, retries_used: int) -> bool:
        if self.unbounded:
            return False
        return retries_used >= int(self.limit)


def decode_fallback(action) -> dict[str, object] | None:
    """Return the declared fallback in the action's own output namespace.

    Stored as JSON so the action stays hashable and its fingerprint stays
    stable, and so that declaring an unserialisable fallback fails when the
    workflow is written rather than when the model first misbehaves.
    """

    fallback_json = getattr(action, "fallback_json", None)
    if fallback_json is None:
        return None
    values = loads_value(fallback_json)
    assert isinstance(values, dict)
    return {name: values[name] for name, _declared in action.outputs}


def checked_llm_outputs(action, named_outputs) -> dict:
    """Check one response against the declared outputs, in their own namespace.

    A missing key or a wrong type is the model's failure, not the workflow's,
    so it is raised as an invalid response and is worth another sample.
    """

    if not isinstance(named_outputs, dict):
        raise LLMInvalidResponseError(
            f"LLM action {action.name!r} returned "
            f"{type(named_outputs).__name__}, not a mapping of outputs."
        )
    checked: dict[str, object] = {}
    for name, declared in action.outputs:
        if name not in named_outputs:
            raise LLMInvalidResponseError(
                f"LLM action {action.name!r} did not return output {name!r}."
            )
        try:
            checked[name] = validate_zvalue(
                named_outputs[name],
                declared,
                context=f"LLM action {action.name!r} output {name!r}",
            )
        except TypeError as exc:
            raise LLMInvalidResponseError(str(exc)) from exc
    return checked


def attempt_llm_action(
    action,
    call: Callable[[], dict[str, object]],
    *,
    stop: threading.Event | None = None,
    report: Callable[[str], None] | None = None,
    check: Callable[[object], dict[str, object]] | None = None,
) -> dict[str, object]:
    """Run one LLM action to a usable answer, a declared fallback, or an error.

    ``call`` performs one complete attempt: request, parse, coerce and check
    against the declared outputs. It raises one of the three errors above, or
    returns the outputs. Everything else it raises is a defect and travels
    straight out.

    The counter lives on this stack frame, so two lifelines calling the same
    backend at the same time cannot share or exhaust each other's budget.
    """

    budget = _Budget(getattr(action, "retries", 0))
    fallback = decode_fallback(action)
    if fallback is not None and check is not None:
        # The fallback takes the same road as a real answer, so a declared
        # tuple stays a tuple and nothing reaches durable state unchecked.
        fallback = check(fallback)
    retries_used = 0

    while True:
        try:
            return call()
        except LLMPermanentError as exc:
            # No number of attempts changes this answer. The fallback is the
            # only thing that can, and only if the workflow asked for one.
            if fallback is not None:
                _say(report, f"{action.name}: {exc}. Using the declared fallback.")
                return dict(fallback)
            raise
        except (LLMTransientError, LLMInvalidResponseError) as exc:
            if budget.exhausted(retries_used):
                if fallback is not None:
                    _say(
                        report,
                        f"{action.name}: giving up after {retries_used} "
                        f"retries. Using the declared fallback.",
                    )
                    return dict(fallback)
                raise
            retry_after = getattr(exc, "retry_after", None)
            delay = retry_delays(retry_after, retries_used)
            _say(
                report,
                f"{action.name}: {_one_line(exc)} — retrying in "
                f"{delay:.0f}s ({_remaining(budget, retries_used)}).",
            )
            _sleep_or_cancel(delay, stop)
            retries_used += 1


def _remaining(budget: _Budget, retries_used: int) -> str:
    if budget.unbounded:
        return f"attempt {retries_used + 2}, no limit"
    return f"{int(budget.limit) - retries_used} left"


def _one_line(exc: BaseException, limit: int = 160) -> str:
    """One line, so a long retry never becomes a wall of repeated tracebacks."""

    text = " ".join(str(exc).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _say(report: Callable[[str], None] | None, message: str) -> None:
    if report is not None:
        report(message)
