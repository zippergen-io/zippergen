"""Inspect static role templates and fresh-participant allocation choices."""
from examples.dynamic_spawn.workflow import onboard_employees
from zippergen.semantic import workflow_semantics
from zippergen.view import ViewOptions, render_workflow, workflow_view_data


def test_global_view_keeps_allocation_source_parameters():
    code = render_workflow(onboard_employees)
    assert "at=HR" in code
    assert "retain=(HR,)" in code
    assert "Employee: response = confirm(details)" in code
    assert "bindings=" not in code
    assert "send_descriptor" not in code
    assert "send_invitation" not in code


def test_fresh_role_view_shows_its_child_and_no_parent_loop():
    code = render_workflow(onboard_employees, options=ViewOptions(agent="Employee"))
    assert "no program in the initial session" in code
    assert "@session_role('Employee'" in code
    assert "fresh endpoint" in code
    assert "details = recv('HR')" in code
    assert "send('HR', response)" in code
    assert "while" not in code
    data = workflow_view_data(onboard_employees, options=ViewOptions(agent="Employee"))
    assert set(data["lifelines"]) == {"HR", "Employee"}


def test_initiator_local_view_shows_creation_before_fork():
    hr = render_workflow(onboard_employees, options=ViewOptions(agent="HR"))
    assert hr.index("allocate_session(") < hr.index("send_invitation(")
    assert hr.index("send_invitation(") < hr.index("with fork_session(")
    assert "confirmation = recv('Employee')" in hr


def test_semantics_records_allocation_and_parent_role_boundary():
    model = workflow_semantics(onboard_employees)
    assert set(model["lifelines"]) == {"HR", "Employee"}
    assert model["root_lifelines"] == ["HR"]
    region = next(region for region in model["regions"] if region["kind"] == "spawn")
    assert region["initiator"] == "HR"
    assert region["retain"] == ["HR"]
    assert region["allocate"] == ["Employee"]
    assert set(region["fresh"]) == {"HR.hire", "HR.confirmation"}
