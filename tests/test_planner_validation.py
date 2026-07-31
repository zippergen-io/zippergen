"""Tests for the _validate_planner_spec structural validator."""

from zippergen.runtime import _validate_planner_spec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CALLER = "Planner"
KNOWN = {"write": 1, "critique": 1, "refine": 1}


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
    # Both branches send a value to Planner under the same name (`result`),
    # so `result` is guaranteed available on all paths.
    return """\
@workflow
def generated_workflow(text: str @ Planner) -> str:
    Planner(text) >> Worker1(text)
    Worker1: (draft, needs_revision) = write(text)
    if needs_revision @ Worker1:
        Worker1(draft) >> Worker2(draft)
        Worker2: result = critique(draft)
        Worker2(result) >> Planner(result)
    else:
        Worker1(draft) >> Planner(result)
    return result @ Planner
"""


# ---------------------------------------------------------------------------
# Valid cases
# ---------------------------------------------------------------------------

def test_valid_linear():
    result = _validate_planner_spec(_linear_spec(), CALLER, KNOWN)
    assert result is None


def test_valid_if_branch():
    result = _validate_planner_spec(_if_spec(), CALLER, {"write": 2, "critique": 1})
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
# Scope: lifeline cannot use variables it has not received
# ---------------------------------------------------------------------------

def test_worker_uses_unreceived_variable():
    # Worker1 tries to send `text` but has never received it — scope error.
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
    assert "Worker1" in result


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


def test_return_to_non_caller_lifeline_is_valid():
    # Returning from any lifeline is allowed — `draft` is in scope on Worker2.
    spec = """\
@workflow
def generated_workflow(text: str @ Planner) -> str:
    Planner(text) >> Worker1(text)
    Worker1: draft = write(text)
    Worker1(draft) >> Worker2(draft)
    return draft @ Worker2
"""
    result = _validate_planner_spec(spec, CALLER, KNOWN)
    assert result is None


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
    result = _validate_planner_spec(spec, CALLER, {"write": 1, "refine": 1})
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
    result = _validate_planner_spec(_if_spec(), CALLER, {"write": 2, "critique": 1})
    assert result is None


def test_return_var_only_in_one_branch():
    # True branch sends `final`, false branch sends `draft` — different names.
    # `draft` is not available on all paths, so `return draft @ Planner` must fail.
    spec = """\
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
    result = _validate_planner_spec(spec, CALLER, {"write": 2, "critique": 1})
    assert result is not None
    assert "draft" in result or "path" in result.lower() or "branch" in result.lower()


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
    result = _validate_planner_spec(spec, CALLER, {"write": 2, "critique": 1})
    assert result is not None
    # Worker2 used `secret` but never received it
    assert "secret" in result or "Worker2" in result


# ---------------------------------------------------------------------------
# Self-sends
# ---------------------------------------------------------------------------

def test_self_send_valid():
    # Worker1 renames `draft` to `result` via self-send — distinct names, valid.
    spec = """\
@workflow
def generated_workflow(text: str @ Planner) -> str:
    Planner(text) >> Worker1(text)
    Worker1: draft = write(text)
    Worker1(draft) >> Worker1(result)
    Worker1(result) >> Planner(result)
    return result @ Planner
"""
    result = _validate_planner_spec(spec, CALLER, KNOWN)
    assert result is None


def test_self_send_overlap_rejected():
    # Worker1(draft) >> Worker1(draft) is a no-op — same name on both sides.
    spec = """\
@workflow
def generated_workflow(text: str @ Planner) -> str:
    Planner(text) >> Worker1(text)
    Worker1: draft = write(text)
    Worker1(draft) >> Worker1(draft)
    Worker1(draft) >> Planner(draft)
    return draft @ Planner
"""
    result = _validate_planner_spec(spec, CALLER, KNOWN)
    assert result is not None
    assert "no-op" in result or "same variable" in result or "overlap" in result.lower()


def test_unknown_lifeline_rejected_when_allowed_set_given():
    spec = """\
@workflow
def generated_workflow(text: str @ Planner) -> str:
    Planner(text) >> Worker1(text)
    Worker1: draft = write(text)
    Worker1(draft) >> Worker9(draft)
    return draft @ Worker9
"""
    result = _validate_planner_spec(
        spec,
        CALLER,
        KNOWN,
        {"text": str},
        str,
        {"Planner", "Worker1", "Worker2"},
    )
    assert result is not None
    assert "Unknown lifeline" in result
    assert "Worker9" in result


def test_nested_action_argument_rejected():
    spec = """\
@workflow
def generated_workflow(text: str @ Planner) -> str:
    Planner(text) >> Worker1(text)
    Worker1: draft = refine(write(text))
    Worker1(draft) >> Planner(draft)
    return draft @ Planner
"""
    result = _validate_planner_spec(spec, CALLER, {"write": 1, "refine": 1})
    assert result is not None
    assert "nested action calls" in result


def test_condition_call_rejected():
    spec = """\
@workflow
def generated_workflow(text: str @ Planner) -> str:
    Planner(text) >> Worker1(text)
    Worker1: draft = write(text)
    if critique(draft) @ Worker1:
        Worker1(draft) >> Planner(result)
    else:
        Worker1(draft) >> Planner(result)
    return result @ Planner
"""
    result = _validate_planner_spec(spec, CALLER, {"write": 1, "critique": 1})
    assert result is not None
    assert "Condition" in result
    assert "function call" in result


def test_branch_return_rejected():
    spec = """\
@workflow
def generated_workflow(text: str @ Planner) -> str:
    Planner(text) >> Worker1(text)
    Worker1: draft = write(text)
    if draft @ Worker1:
        return draft @ Worker1
    else:
        Worker1(draft) >> Planner(draft)
    return draft @ Planner
"""
    result = _validate_planner_spec(spec, CALLER, KNOWN)
    assert result is not None
    assert "exactly one return" in result


def test_literal_return_rejected():
    spec = """\
@workflow
def generated_workflow(text: str @ Planner) -> float:
    Planner(text) >> Worker1(text)
    Worker1: draft = write(text)
    return 0.0 @ Planner
"""
    result = _validate_planner_spec(spec, CALLER, KNOWN, {"text": str}, float)
    assert result is not None
    assert "return var @ Lifeline" in result


# ---------------------------------------------------------------------------
# Generated-code safety boundary
# ---------------------------------------------------------------------------

def _generated_llm_spec(*, system: str = '"Draft safely."') -> str:
    return f'''\
@llm(
    system={system},
    user="{{text}}",
    parse="text",
    outputs=(("draft", str),),
)
def draft(text: str): ...

@workflow
def generated_workflow(text: str @ Planner) -> str:
    Planner(text) >> Worker1(text)
    Worker1: draft_value = draft(text)
    Worker1(draft_value) >> Planner(draft_value)
    return draft_value @ Planner
'''


def test_safe_generated_llm_definition_is_accepted():
    result = _validate_planner_spec(
        _generated_llm_spec(),
        CALLER,
        {},
        {"text": str},
        str,
        {"Planner", "Worker1"},
        {"llm"},
    )
    assert result is None


def test_module_import_is_rejected_before_execution():
    result = _validate_planner_spec(
        "import os\n" + _linear_spec(),
        CALLER,
        KNOWN,
    )
    assert result is not None
    assert "Unsafe generated planner code" in result
    assert "module-level code" in result


def test_module_call_is_rejected_before_execution():
    result = _validate_planner_spec(
        'open("marker", "w").write("owned")\n' + _linear_spec(),
        CALLER,
        KNOWN,
    )
    assert result is not None
    assert "Unsafe generated planner code" in result
    assert "module-level code" in result


def test_workflow_body_python_call_is_rejected_before_execution():
    spec = '''\
@workflow
def generated_workflow(text: str @ Planner) -> str:
    open("marker", "w").write("owned")
    Planner(text) >> Worker1(text)
    Worker1: draft = write(text)
    Worker1(draft) >> Planner(draft)
    return draft @ Planner
'''
    result = _validate_planner_spec(spec, CALLER, KNOWN)
    assert result is not None
    assert "Unsafe generated planner code" in result
    assert "workflow expressions" in result


def test_generated_default_expression_is_rejected_before_execution():
    spec = '''\
@workflow
def generated_workflow(
    text: str @ Planner = open("marker", "w").write("owned"),
) -> str:
    Planner(text) >> Worker1(text)
    Worker1: draft = write(text)
    Worker1(draft) >> Planner(draft)
    return draft @ Planner
'''
    result = _validate_planner_spec(spec, CALLER, KNOWN)
    assert result is not None
    assert "Unsafe generated planner code" in result
    assert "required positional parameters" in result


def test_generated_llm_configuration_must_be_constant():
    result = _validate_planner_spec(
        _generated_llm_spec(system='open("marker", "w").write("owned")'),
        CALLER,
        {},
        {"text": str},
        str,
        {"Planner", "Worker1"},
        {"llm"},
    )
    assert result is not None
    assert "Unsafe generated planner code" in result
    assert "system= must be a constant string" in result


def test_generated_llm_body_must_be_ellipsis():
    spec = _generated_llm_spec().replace(
        "def draft(text: str): ...",
        'def draft(text: str):\n    return open("marker", "w").write("owned")',
    )
    result = _validate_planner_spec(
        spec,
        CALLER,
        {},
        {"text": str},
        str,
        {"Planner", "Worker1"},
        {"llm"},
    )
    assert result is not None
    assert "Unsafe generated planner code" in result
    assert "body must be exactly `...`" in result


def test_generated_pure_action_is_rejected():
    spec = '''\
@pure
def draft(text: str) -> str:
    return open("marker", "w").write("owned")

@workflow
def generated_workflow(text: str @ Planner) -> str:
    Planner(text) >> Worker1(text)
    Worker1: draft_value = draft(text)
    Worker1(draft_value) >> Planner(draft_value)
    return draft_value @ Planner
'''
    result = _validate_planner_spec(
        spec,
        CALLER,
        {},
        {"text": str},
        str,
        {"Planner", "Worker1"},
        {"llm"},
    )
    assert result is not None
    assert "Unsafe generated planner code" in result
    assert "generated @pure actions are disabled" in result


def test_generated_llm_requires_explicit_allow():
    result = _validate_planner_spec(
        _generated_llm_spec(),
        CALLER,
        {},
        {"text": str},
        str,
        {"Planner", "Worker1"},
        set(),
    )
    assert result is not None
    assert "Unsafe generated planner code" in result
    assert "generated @llm actions are not enabled" in result


def test_generated_action_cannot_replace_predefined_action():
    spec = _generated_llm_spec().replace("def draft(", "def write(").replace(
        "draft(text)",
        "write(text)",
    )
    result = _validate_planner_spec(
        spec,
        CALLER,
        {"write": 1},
        {"text": str},
        str,
        {"Planner", "Worker1"},
        {"llm"},
    )
    assert result is not None
    assert "Unsafe generated planner code" in result
    assert "may not replace trusted bindings: write" in result


def test_generated_control_flow_requires_explicit_allow():
    result = _validate_planner_spec(
        _if_spec(),
        CALLER,
        {"write": 2, "critique": 1},
        {"text": str},
        str,
        {"Planner", "Worker1", "Worker2"},
        set(),
    )
    assert result is not None
    assert "Unsafe generated planner code" in result
    assert "generated if control flow is not enabled" in result
