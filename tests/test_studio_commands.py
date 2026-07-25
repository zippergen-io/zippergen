from zippergen.studio_commands import (
    COMMANDS,
    command_spec,
    full_help,
    natural_command_catalog,
    subcommand_completions,
    top_level_completions,
)


def test_command_registry_has_unique_paths_and_derives_user_surfaces():
    paths = [command.path for command in COMMANDS]

    assert len(paths) == len(set(paths))
    assert ("workflow", "prompts") not in paths
    assert ("studio", "doctor") in paths
    assert ("studio", "restart") in paths
    assert "studio doctor" in full_help()
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
    catalog = natural_command_catalog()
    assert "workflow discard" in catalog
    assert "studio doctor" in catalog
    assert "\n- exit" not in catalog
