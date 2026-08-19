"""Model routing decisions."""

from zippergen.models import fake_model_notice


def test_a_run_says_which_participants_answer_with_the_mock():
    """A mostly-fake run must not look like a real one.

    `zg init` sets `default = "mock"` so a new project runs before anybody
    owns an API key. That is right; silence about it is not.
    """

    from zippergen.models import fake_model_notice

    some = fake_model_notice({"Writer": "mock", "Editor": "openai:gpt-4o"})
    every = fake_model_notice({"Writer": "mock", "Editor": "mock"})
    none = fake_model_notice({"Editor": "openai:gpt-4o"})

    assert some is not None and "Writer" in some and "Editor" not in some
    assert every is not None and "No real model is in use" in every
    assert none is None, "a fully real run should say nothing"
    assert "zg model assign" in some
