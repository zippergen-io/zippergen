import json

from zippergen import (
    DeploymentField,
    DeploymentSpec,
    Lifeline,
    Var,
    effect,
    pure,
    workflow,
)
from zippergen.view import ViewOptions, render_workflow, render_workflow_json


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


zippergen_deployment = DeploymentSpec(
    name="editorial",
    fields=(DeploymentField("channel", "Publishing channel", default="draft"),),
)


def test_global_protocol_view_is_source_like_code():
    code = render_workflow(editorial, options=ViewOptions())

    assert "@workflow" in code
    assert "def editorial(text: str @ Author, approved: bool @ Author) -> str:" in code
    assert "Author(text) >> Editor(text)" in code
    assert "Editor: edited = edit_text(text)" in code
    assert "if (approved) @ Author:" in code
    assert "return published @ Publisher" in code


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
