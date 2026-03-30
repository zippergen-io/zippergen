"""Tests for the _validate_planner_spec structural validator."""

from zippergen.runtime import _validate_planner_spec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CALLER = "Planner"
KNOWN = {"write", "critique", "refine"}


def _linear_spec():
    return """\
@workflow
def generated_workflow(text: str @ Planner, instructions: str @ Planner) -> str:
    Planner(text, instructions) >> Worker1(text, instructions)
    Worker1: draft = write(text, instructions)
    Worker1(draft) >> Planner(draft)
    return draft @ Planner
"""


def _if_spec():
    return """\
@workflow
def generated_workflow(text: str @ Planner) -> str:
    Planner(text) >> Worker1(text)
    Worker1: (draft, needs_revision) = write(text)
    if needs_revision @ Worker1:
        Worker1(draft) >> Worker2(draft)
        Worker2: final = critique(draft)
        Worker2(final) >> Planner(final)
    else:
        Worker1(draft) >> Planner(draft)
    return draft @ Planner
"""


# ---------------------------------------------------------------------------
# Valid cases
# ---------------------------------------------------------------------------

def test_valid_linear():
    result = _validate_planner_spec(_linear_spec(), CALLER, KNOWN)
    assert result is None


def test_valid_if_branch():
    result = _validate_planner_spec(_if_spec(), CALLER, {"write", "critique"})
    assert result is None


# ---------------------------------------------------------------------------
# Invariant: missing generated_workflow function
# ---------------------------------------------------------------------------

def test_missing_generated_workflow():
    spec = """\
@workflow
def some_other_function(text: str @ Planner) -> str:
    Planner(text) >> Worker1(text)
    Worker1: draft = write(text)
    Worker1(draft) >> Planner(draft)
    return draft @ Planner
"""
    result = _validate_planner_spec(spec, CALLER, KNOWN)
    assert result is not None
    assert "generated_workflow" in result


# ---------------------------------------------------------------------------
# Invariant 1: first statement must send FROM caller
# ---------------------------------------------------------------------------

def test_first_statement_not_from_caller():
    spec = """\
@workflow
def generated_workflow(text: str @ Planner) -> str:
    Worker1(text) >> Worker2(text)
    Worker2: draft = write(text)
    Worker2(draft) >> Planner(draft)
    return draft @ Planner
"""
    result = _validate_planner_spec(spec, CALLER, KNOWN)
    assert result is not None
    assert CALLER in result


# ---------------------------------------------------------------------------
# Invariant 3: last statement must be `return var @ caller`
# ---------------------------------------------------------------------------

def test_last_statement_not_return():
    spec = """\
@workflow
def generated_workflow(text: str @ Planner) -> str:
    Planner(text) >> Worker1(text)
    Worker1: draft = write(text)
    Worker1(draft) >> Planner(draft)
"""
    result = _validate_planner_spec(spec, CALLER, KNOWN)
    assert result is not None


def test_last_return_wrong_lifeline():
    spec = """\
@workflow
def generated_workflow(text: str @ Planner) -> str:
    Planner(text) >> Worker1(text)
    Worker1: draft = write(text)
    Worker1(draft) >> Worker2(draft)
    return draft @ Worker2
"""
    result = _validate_planner_spec(spec, CALLER, KNOWN)
    assert result is not None
    assert CALLER in result


# ---------------------------------------------------------------------------
# Invariant 2: second-to-last statement must send TO caller
# ---------------------------------------------------------------------------

def test_second_to_last_sends_to_wrong_lifeline():
    spec = """\
@workflow
def generated_workflow(text: str @ Planner) -> str:
    Planner(text) >> Worker1(text)
    Worker1: draft = write(text)
    Worker1(draft) >> Worker2(draft)
    return draft @ Planner
"""
    result = _validate_planner_spec(spec, CALLER, KNOWN)
    assert result is not None
    assert CALLER in result


# ---------------------------------------------------------------------------
# Invariant 4: unknown action called
# ---------------------------------------------------------------------------

def test_unknown_action():
    spec = """\
@workflow
def generated_workflow(text: str @ Planner) -> str:
    Planner(text) >> Worker1(text)
    Worker1: draft = unknown_action(text)
    Worker1(draft) >> Planner(draft)
    return draft @ Planner
"""
    result = _validate_planner_spec(spec, CALLER, KNOWN)
    assert result is not None
    assert "unknown_action" in result


# ---------------------------------------------------------------------------
# Invariant 5: mismatched >> arg counts
# ---------------------------------------------------------------------------

def test_mismatched_rshift_arg_counts():
    spec = """\
@workflow
def generated_workflow(text: str @ Planner, instructions: str @ Planner) -> str:
    Planner(text, instructions) >> Worker1(text)
    Worker1: draft = write(text)
    Worker1(draft) >> Planner(draft)
    return draft @ Planner
"""
    result = _validate_planner_spec(spec, CALLER, KNOWN)
    assert result is not None
    assert "mismatched" in result.lower() or "mismatch" in result.lower() or "arg" in result.lower() or "count" in result.lower()


# ---------------------------------------------------------------------------
# Invariant 6: lifeline uses variable it never received
# ---------------------------------------------------------------------------

def test_lifeline_uses_unreceived_var():
    # Worker2 never receives `text` but tries to use it
    spec = """\
@workflow
def generated_workflow(text: str @ Planner, instructions: str @ Planner) -> str:
    Planner(text, instructions) >> Worker1(text, instructions)
    Worker1: draft = write(text, instructions)
    Worker1(draft) >> Worker2(draft)
    Worker2: final = refine(draft, text)
    Worker2(final) >> Planner(final)
    return final @ Planner
"""
    result = _validate_planner_spec(spec, CALLER, {"write", "refine"})
    assert result is not None
    assert "text" in result or "Worker2" in result


def test_send_uses_var_sender_doesnt_have():
    # Worker1 tries to send `instructions` but never received it
    spec = """\
@workflow
def generated_workflow(text: str @ Planner, instructions: str @ Planner) -> str:
    Planner(text) >> Worker1(text)
    Worker1: draft = write(text)
    Worker1(draft, instructions) >> Planner(draft, instructions)
    return draft @ Planner
"""
    result = _validate_planner_spec(spec, CALLER, KNOWN)
    assert result is not None
    assert "instructions" in result or "Worker1" in result


# ---------------------------------------------------------------------------
# if-branch scoping
# ---------------------------------------------------------------------------

def test_if_branch_correct_scoping():
    result = _validate_planner_spec(_if_spec(), CALLER, {"write", "critique"})
    assert result is None


def test_if_branch_inner_lifeline_uses_unreceived_var():
    spec = """\
@workflow
def generated_workflow(text: str @ Planner, secret: str @ Planner) -> str:
    Planner(text) >> Worker1(text)
    Worker1: (draft, needs_revision) = write(text)
    if needs_revision @ Worker1:
        Worker1(draft) >> Worker2(draft)
        Worker2: final = critique(draft, secret)
        Worker2(final) >> Planner(final)
    else:
        Worker1(draft) >> Planner(draft)
    return draft @ Planner
"""
    result = _validate_planner_spec(spec, CALLER, {"write", "critique"})
    assert result is not None
    # Worker2 used `secret` but never received it
    assert "secret" in result or "Worker2" in result
