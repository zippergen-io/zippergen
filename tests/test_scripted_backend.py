"""Deterministic scripted responses, so every protocol branch is reachable.

`mock` answers every action with one placeholder, so a workflow driven by it
takes whichever branch that placeholder selects and no other. In a consensus
protocol that means immediate agreement, and everything behind a decision is
unreachable. These tests pin the backend that fixes it.
"""

import json
import threading

import pytest

from zippergen.backends import (
    backend_from_spec,
    load_scripted_script,
    make_scripted_backend,
)


class _Action:
    """The parts of an action a backend actually reads."""

    def __init__(self, name, outputs):
        self.name = name
        self.outputs = outputs


ASSESS = _Action("assess", (("verdict", str), ("reason", str)))


def _as(lifeline, backend, action=ASSESS):
    """Call the backend as the runtime does: thread name is the lifeline."""

    captured = {}

    def call():
        try:
            captured["result"] = backend(action, {})
        except BaseException as exc:  # re-raised in the caller below
            captured["error"] = exc

    thread = threading.Thread(target=call, name=lifeline)
    thread.start()
    thread.join()
    if "error" in captured:
        raise captured["error"]
    return captured["result"]


def test_each_participant_gets_its_own_answer():
    backend = make_scripted_backend(
        {
            "LLM1.assess": [{"verdict": "yes", "reason": "a"}],
            "LLM2.assess": [{"verdict": "no", "reason": "b"}],
        }
    )

    assert _as("LLM1", backend)["verdict"] == "yes"
    assert _as("LLM2", backend)["verdict"] == "no"


def test_responses_are_consumed_in_order():
    backend = make_scripted_backend(
        {"LLM1.assess": [
            {"verdict": "no", "reason": "first"},
            {"verdict": "yes", "reason": "second"},
        ]}
    )

    assert _as("LLM1", backend)["reason"] == "first"
    assert _as("LLM1", backend)["reason"] == "second"


def test_the_last_response_repeats_once_exhausted():
    """A reviewer who never changes its mind is a one-element list.

    Without this, driving a loop to its bound means writing one entry per
    iteration and updating them whenever the bound changes.
    """

    backend = make_scripted_backend(
        {"LLM1.assess": [{"verdict": "no", "reason": "unmoved"}]}
    )

    for _ in range(5):
        assert _as("LLM1", backend)["verdict"] == "no"


def test_a_bare_action_name_answers_for_every_participant():
    backend = make_scripted_backend(
        {"assess": [{"verdict": "maybe", "reason": "shared"}]}
    )

    assert _as("LLM1", backend)["verdict"] == "maybe"
    assert _as("LLM2", backend)["verdict"] == "maybe"


def test_a_participant_key_wins_over_a_bare_action_name():
    backend = make_scripted_backend(
        {
            "assess": [{"verdict": "shared", "reason": "x"}],
            "LLM2.assess": [{"verdict": "specific", "reason": "y"}],
        }
    )

    assert _as("LLM1", backend)["verdict"] == "shared"
    assert _as("LLM2", backend)["verdict"] == "specific"


def test_an_unscripted_action_names_itself_and_what_is_scripted():
    backend = make_scripted_backend({"LLM1.assess": [{"verdict": "y", "reason": "r"}]})

    with pytest.raises(RuntimeError) as error:
        _as("LLM2", backend, _Action("reconsider", (("verdict", str),)))

    message = str(error.value)
    assert "LLM2.reconsider" in message
    assert "LLM1.assess" in message


def test_a_response_missing_a_declared_output_says_which():
    backend = make_scripted_backend({"LLM1.assess": [{"verdict": "yes"}]})

    with pytest.raises(RuntimeError, match="missing reason"):
        _as("LLM1", backend)


def test_extra_keys_in_a_response_are_ignored():
    backend = make_scripted_backend(
        {"LLM1.assess": [{"verdict": "y", "reason": "r", "note": "for humans"}]}
    )

    assert _as("LLM1", backend) == {"verdict": "y", "reason": "r"}


def test_a_single_object_is_accepted_where_a_list_is_expected(tmp_path):
    path = tmp_path / "script.json"
    path.write_text(json.dumps({"LLM1.assess": {"verdict": "y", "reason": "r"}}))

    backend = make_scripted_backend(load_scripted_script(path))

    assert _as("LLM1", backend)["verdict"] == "y"


def test_the_spec_builds_the_backend_from_a_file(tmp_path):
    path = tmp_path / "script.json"
    path.write_text(json.dumps({"assess": [{"verdict": "y", "reason": "r"}]}))

    backend, label = backend_from_spec(f"scripted:{path}")

    assert _as("LLM1", backend)["verdict"] == "y"
    assert str(path) in label


def test_the_spec_requires_a_file():
    with pytest.raises(RuntimeError, match="needs a response file"):
        backend_from_spec("scripted")


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("[1, 2]", "must be a JSON object"),
        ("{not json", "not valid JSON"),
        ('{"assess": ["a string"]}', "must be objects"),
    ],
)
def test_a_malformed_script_is_rejected_when_it_is_read(tmp_path, content, expected):
    path = tmp_path / "script.json"
    path.write_text(content)

    with pytest.raises(RuntimeError, match=expected):
        load_scripted_script(path)


def test_a_missing_file_names_the_path(tmp_path):
    with pytest.raises(RuntimeError, match="not found"):
        load_scripted_script(tmp_path / "absent.json")
