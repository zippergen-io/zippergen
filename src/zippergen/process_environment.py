"""Applying values to the process environment, and taking them back off.

A workflow run, a readiness check, and a foreground command all need to put
provider credentials and connector routing into ``os.environ`` for the length
of one operation and then restore exactly what was there. Three copies of that
context manager existed, and the one in the CLI additionally stringified its
values -- so a readiness check could observe a different environment from the
run it was meant to be checking.

One implementation, one contract: keys and values are strings, absent stays
absent, and what was there before is what is there afterwards.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager


@contextmanager
def temporary_environment(values: Mapping[str, object]) -> Iterator[None]:
    """Apply ``values`` to ``os.environ`` for the duration of the block.

    Values are coerced with ``str`` because the environment holds only
    strings; doing it here means no caller has to remember to.
    """

    previous = {name: os.environ.get(name) for name in values}
    os.environ.update({name: str(value) for name, value in values.items()})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


__all__ = ["temporary_environment"]
