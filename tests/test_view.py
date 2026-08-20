import json

from zippergen import (
    DeploymentField,
    DeploymentSpec,
    Json,
    Lifeline,
    Var,
    effect,
    llm,
    pure,
    workflow,
)
from zippergen.locator import resolve_path, statement_node_paths
from zippergen.projection import project
from zippergen.syntax import ActStmt
from zippergen.view import (
    ViewOptions,
    _render_action,
    render_local_projection_with_pointers,
    render_workflow,
    render_workflow_json,
)


Author = Lifeline("Author")
Editor = Lifeline("Editor")
Publisher = Lifeline("Publisher")

text = Var("text", str)
edited = Var("edited", str)
published = Var("published", str)
approved = Var("approved", bool)


@pure
def edit_text(value: str) -> str:
    return value.strip()


@effect
def publish_text(value: str) -> str:
    return f"published:{value}"


@workflow
def editorial(text: str @ Author, approved: bool @ Author) -> str:
    Author(text) >> Editor(text)
    Editor: edited = edit_text(text)
    if approved @ Author:
        Editor(edited) >> Publisher(edited)
        Publisher: published = publish_text(edited)
    else:
        Author(text) >> Publisher(published)
    return published @ Publisher


@workflow
def structured_handoff(payload: Json @ Author) -> Json:
    Author(payload) >> Editor(payload)
    return payload @ Editor


zippergen_deployment = DeploymentSpec(
    fields=(DeploymentField("channel", "Publishing channel", default="draft"),),
)


@llm(
    system="Decide.",
    user="{value}",
    parse="bool",
    outputs=[("accepted", bool)],
    fallback=False,
)
def decide(value: str): ...


def test_global_protocol_view_is_source_like_code():
    code = render_workflow(editorial, options=ViewOptions())

    assert "@workflow" in code
    assert "def editorial(text: str @ Author, approved: bool @ Author) -> str:" in code
    assert "Author(text) >> Editor(text)" in code
    assert "Editor: edited = edit_text(text)" in code
    assert "if (approved) @ Author:" in code
    assert "return published @ Publisher" in code


def test_llm_fallback_is_visible_in_the_action_view():
    rendered = "\n".join(_render_action(decide, full=True))

    assert "fallback=False" in rendered


def test_json_type_is_visible_in_code_and_structured_views():
    code = render_workflow(structured_handoff, options=ViewOptions())
    data = json.loads(render_workflow_json(structured_handoff))

    assert "def structured_handoff(payload: Json @ Author) -> Json:" in code
    assert data["inputs"][0]["type"] == "Json"
    assert data["outputs"][0]["type"] == "Json"


def test_communications_view_erases_actions_but_keeps_control_structure():
    code = render_workflow(
        editorial,
        options=ViewOptions(communications_only=True),
    )

    assert "Author(text) >> Editor(text)" in code
    assert "if (approved) @ Author:" in code
    assert "Editor(edited) >> Publisher(edited)" in code
    assert "edit_text" not in code
    assert "publish_text" not in code


def test_one_agent_view_is_exact_local_projection_code():
    code = render_workflow(editorial, options=ViewOptions(agent="Editor"))

    assert "Generated local projection for Editor" in code
    assert "text = recv('Author')" in code
    assert "edited = edit_text(text)" in code
    assert "if recv_decision('Author')" in code
    assert "send('Publisher', edited)" in code
    assert "publish_text" not in code


def test_live_agent_view_marks_the_stable_local_statement_path():
    local = project(editorial, Editor)
    paths = statement_node_paths(local)
    action_path = next(
        path
        for path in paths.values()
        if isinstance(resolve_path(local, path), ActStmt)
    )

    code = render_local_projection_with_pointers(
        editorial,
        "Editor",
        [action_path],
    )

    assert "▶     edited = edit_text(text)" in code
    assert "  " + "    text = recv('Author')" in code
    assert code.startswith(
        "# Live local projection for Editor; read-only observation."
    )

    watched = render_local_projection_with_pointers(
        editorial,
        "Editor",
        [action_path],
        indent=1,
    )
    assert " ▶     edited = edit_text(text)" in watched
    assert watched.startswith(
        " # Live local projection for Editor; read-only observation."
    )


def test_selected_agent_view_preserves_hidden_agents_as_boundaries():
    code = render_workflow(
        editorial,
        options=ViewOptions(agents=("Editor", "Publisher")),
    )

    assert "external('Author', text) >> Editor(text)" in code
    assert "if (approved) @ external('Author'):" in code
    assert "Editor(edited) >> Publisher(edited)" in code
    assert "external('Author', text) >> Publisher(published)" in code


def test_full_view_includes_action_implementation_and_deployment_data():
    code = render_workflow(
        editorial,
        module=__import__(__name__),
        options=ViewOptions(detail="full"),
    )

    assert "def edit_text(value: str) -> str:" in code
    assert "zippergen_deployment =" in code
    assert "Publishing channel" in code


def test_json_view_contains_code_and_structured_metadata():
    payload = json.loads(render_workflow_json(editorial, options=ViewOptions(agent="Publisher")))

    assert payload["workflow"] == "editorial"
    assert payload["agent"] == "Publisher"
    assert payload["lifelines"] == ["Author", "Editor", "Publisher"]
    assert "Generated local projection for Publisher" in payload["code"]
