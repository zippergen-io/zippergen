"""A run that is working must not look like a run that has hung.

A model or assistant call is silent for minutes, and silence is
indistinguishable from a hang. These pin the two rules that keep the indicator
from causing harm while it removes that ambiguity.
"""

import io
import time

from zippergen.activity import ActivityIndicator, format_duration


class _Terminal(io.StringIO):
    """A stream that claims to be a terminal, so drawing is enabled."""

    def isatty(self) -> bool:
        return True


def _drawn(stream: io.StringIO) -> str:
    return stream.getvalue()


def test_it_says_what_is_running(monkeypatch):
    stream = _Terminal()
    indicator = ActivityIndicator(stream, interactive=True)

    indicator.started(1, "Implementer · implement")
    deadline = time.time() + 2
    while time.time() < deadline and "implement" not in _drawn(stream):
        time.sleep(0.02)
    indicator.close()

    assert "Implementer · implement" in _drawn(stream)


def test_it_stays_silent_while_a_person_is_answering():
    """A prompt is read from the same terminal; drawing under it corrupts input."""

    stream = _Terminal()
    indicator = ActivityIndicator(stream, interactive=True)

    indicator.person_waiting(True)
    indicator.started(1, "Reviewer · review")
    time.sleep(0.4)
    during = _drawn(stream)
    indicator.close()

    assert during == "", "nothing may be drawn while a person is being asked"


def test_it_resumes_once_the_person_has_answered():
    stream = _Terminal()
    indicator = ActivityIndicator(stream, interactive=True)
    indicator.started(1, "Reviewer · review")
    indicator.person_waiting(True)
    time.sleep(0.3)

    indicator.person_waiting(False)
    deadline = time.time() + 2
    while time.time() < deadline and "review" not in _drawn(stream):
        time.sleep(0.02)
    indicator.close()

    assert "Reviewer · review" in _drawn(stream)


def test_a_redirected_stream_is_left_completely_alone():
    """Piped output, a deployment log and a test all get nothing."""

    stream = io.StringIO()
    indicator = ActivityIndicator(stream, interactive=False)

    indicator.started(1, "Implementer · implement")
    time.sleep(0.3)
    indicator.close()

    assert _drawn(stream) == ""


def test_the_line_is_erased_when_the_work_finishes():
    stream = _Terminal()
    indicator = ActivityIndicator(stream, interactive=True)
    indicator.started(1, "Implementer · implement")
    deadline = time.time() + 2
    while time.time() < deadline and "implement" not in _drawn(stream):
        time.sleep(0.02)

    indicator.finished(1)
    indicator.close()

    assert _drawn(stream).endswith("\r"), "the terminal gets its line back"


def test_elapsed_time_reads_the_way_a_person_says_it():
    assert format_duration(1500) == "1.5s"
    assert format_duration(59_000) == "59.0s"
    assert format_duration(72_000) == "1m12s"
    assert format_duration(3_600_000) == "60m00s"
