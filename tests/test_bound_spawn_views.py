"""Inspect role templates without inventing parent programs for recruits."""
from examples.dynamic_spawn.workflow import access_reviews
from zippergen.semantic import workflow_semantics
from zippergen.view import ViewOptions, render_workflow, workflow_view_data


def test_global_view_keeps_bound_spawn_source_parameters():
    code = render_workflow(access_reviews)
    assert "bindings=bindings @ Manager" in code
    assert "retain=(Manager, IT,)" in code
    assert "Reviewer: approved = assess(request)" in code
    assert "send_descriptor" not in code
    assert "send_invitation" not in code


def test_recruited_role_view_shows_its_child_and_no_parent_loop():
    code = render_workflow(access_reviews, options=ViewOptions(agent="Reviewer"))
    assert "no program in the initial session" in code
    assert "@session_role('Reviewer'" in code
    assert "request = recv('Manager')" in code
    assert "send_decision('IT', True)" in code
    assert "while" not in code
    data = workflow_view_data(access_reviews, options=ViewOptions(agent="Reviewer"))
    assert set(data["lifelines"]) == {"Manager", "Reviewer", "IT"}


def test_bound_owner_and_retained_local_views_show_creation_order():
    manager = render_workflow(access_reviews, options=ViewOptions(agent="Manager"))
    assert manager.index("select_binding(") < manager.index("send_descriptor(")
    assert manager.index("send_invitation(") < manager.index("with fork_session(")
    it = render_workflow(access_reviews, options=ViewOptions(agent="IT"))
    assert it.index("receive_descriptor(") < it.index("with fork_session(")
    assert "recv_decision('Reviewer')" in it


def test_semantics_records_binding_and_parent_role_boundary():
    model = workflow_semantics(access_reviews)
    assert set(model["lifelines"]) == {"Manager", "Reviewer", "IT"}
    assert model["root_lifelines"] == ["Manager", "IT"]
    region = next(region for region in model["regions"] if region["kind"] == "spawn")
    assert region["binding"] == {"owner": "Manager", "expression": "bindings"}
    assert region["retain"] == ["IT", "Manager"]
    assert all(not field.startswith("Reviewer.") for field in region["fresh"])
