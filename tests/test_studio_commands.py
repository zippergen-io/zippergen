from zippergen.studio_commands import (
    COMMANDS,
    WORKFLOW_VIEWS,
    command_spec,
    full_help,
    natural_command_catalog,
    subcommand_completions,
    top_level_completions,
    workflow_view_completions,
    workflow_view_spec,
)


def test_command_registry_has_unique_paths_and_derives_user_surfaces():
    paths = [command.path for command in COMMANDS]

    assert len(paths) == len(set(paths))
    assert ("workflow", "prompts") not in paths
    assert ("studio", "doctor") in paths
    assert ("studio", "restart") in paths
    assert ("store", "list") in paths
    assert ("store", "delete") in paths
    assert ("deployment", "show") in paths
    assert "studio doctor" in full_help()
    assert "store list" in full_help()
    assert "deployment show" in full_help()
    assert "\n  status " not in full_help()
    assert "workflow prompts" not in full_help()
    assert ("studio", "inspect or operate the Studio process") in (
        top_level_completions()
    )
    assert ("doctor", "check local Studio readiness") in (
        subcommand_completions("studio")
    )


def test_command_registry_owns_risk_and_natural_language_catalog():
    assert command_spec(["workflow", "discard"]).risk == "destructive"
    assert command_spec(["workflow", "implement"]).risk == "execution"
    assert command_spec(["workflow", "show", "protocol"]).risk == "read-only"
    assert command_spec(["store", "delete", "demo"]).risk == "destructive"
    catalog = natural_command_catalog()
    assert "workflow discard" in catalog
    assert "studio doctor" in catalog
    assert "\n- exit" not in catalog


def test_workflow_view_registry_owns_commands_labels_aliases_and_completion():
    assert [name for name, _description in workflow_view_completions()] == [
        view.command for view in WORKFLOW_VIEWS
    ]
    communications = workflow_view_spec("communications only")
    complete = workflow_view_spec("Complete workflow")
    one = workflow_view_spec("one participant")
    many = workflow_view_spec("selected participants")
    assert communications is not None and communications.command == "communications"
    assert complete is not None and complete.command == "full"
    assert one is not None and one.participants == "one"
    assert many is not None and many.participants == "many"
    assert workflow_view_spec("not-a-view") is None
