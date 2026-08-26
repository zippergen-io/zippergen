"""Every declared connector kind must be servable and configurable.

A workflow may name any kind ``connectors.CONNECTOR_KINDS`` accepts. Accepting
one that no provider serves produces a workflow that validates, deploys, and
then cannot be configured -- the failure lands on the operator, far from the
line that caused it. These tests hold the declaration and the support together
so a kind can only be added once it is actually supported end to end.
"""

from zippergen.configuration_rendering import CONNECTOR_ROUTE_KINDS
from zippergen.connectors import (
    CONNECTOR_KIND_SPECS,
    CONNECTOR_KINDS,
    ConnectorRequirement,
)
from zippergen.provider_connections import _CONNECTOR_KINDS
from zippergen.serve import _parse_cli_args

import pytest


@pytest.mark.parametrize("spec", CONNECTOR_KIND_SPECS, ids=lambda s: s.name)
def test_every_declared_kind_has_a_provider_that_serves_it(spec) -> None:
    from zippergen.provider_connections import (
        connector_kinds_for_provider,
        provider_supports_connector,
    )

    assert spec.name in connector_kinds_for_provider(spec.provider)
    assert provider_supports_connector(spec.provider, spec.name)


@pytest.mark.parametrize("spec", CONNECTOR_KIND_SPECS, ids=lambda s: s.name)
def test_every_declared_kind_is_a_complete_adapter(spec) -> None:
    """A name is not support: configuration, wiring and checks must come with it."""

    assert spec.settings
    assert spec.credential_environment_field
    assert callable(spec.describe)
    assert callable(spec.readiness)
    assert all(
        setting.name and setting.help and setting.metavar
        for setting in spec.settings
    )


def test_registry_names_and_cli_settings_are_unambiguous() -> None:
    from zippergen.connectors import CONNECTOR_SETTING_SPECS

    assert len(set(CONNECTOR_KINDS)) == len(CONNECTOR_KINDS)
    names = [setting.name for setting in CONNECTOR_SETTING_SPECS]
    assert len(set(names)) == len(names)


def test_generic_connector_consumers_do_not_branch_on_kind_names() -> None:
    """Those modules must consume the adapter, not grow parallel registries."""

    import pathlib
    import re

    source_root = pathlib.Path(__file__).resolve().parents[1] / "src" / "zippergen"
    branch = re.compile(
        r"(?:requirement\.)?kind\s*(?:==|!=)\s*[\"'](?:"
        + "|".join(re.escape(name) for name in CONNECTOR_KINDS)
        + r")[\"']"
    )
    offenders = [
        name
        for name in ("serve.py", "workspace.py")
        if branch.search((source_root / name).read_text())
    ]
    assert not offenders, (
        "generic connector consumers branch on a kind instead of reading its "
        f"adapter: {offenders}"
    )


@pytest.mark.parametrize("spec", CONNECTOR_KIND_SPECS, ids=lambda s: s.name)
def test_every_declared_kind_names_a_credential_its_provider_stores(spec) -> None:
    """A kind whose provider cannot hold its credential cannot be wired."""

    from zippergen.provider_connections import provider_credential_field

    assert provider_credential_field(spec.provider) == spec.credential


@pytest.mark.parametrize("spec", CONNECTOR_KIND_SPECS, ids=lambda s: s.name)
def test_every_declared_kind_can_be_authorized(spec) -> None:
    """A kind with scopes must resolve one for every access level, and a
    kind without scopes must be refused rather than silently unscoped."""

    from zippergen.google_auth import google_scope_for_access

    for access in ("read-only", "write", "read-write"):
        if spec.scopes:
            assert google_scope_for_access(spec.name, access).startswith("https://")
        else:
            with pytest.raises(ValueError):
                google_scope_for_access(spec.name, access)


@pytest.mark.parametrize("spec", CONNECTOR_KIND_SPECS, ids=lambda s: s.name)
def test_every_declared_kind_pulls_in_the_extra_it_needs(spec) -> None:
    """A deployment binding this kind must install what it imports."""

    from zippergen.deployment_environment import _deployment_zippergen_extras

    profile = {"connectors": {"c": {"kind": spec.name}}}
    extras = _deployment_zippergen_extras(profile)
    assert extras == ((spec.extra,) if spec.extra else ())


@pytest.mark.parametrize("spec", CONNECTOR_KIND_SPECS, ids=lambda s: s.name)
def test_every_declared_kind_is_offered_by_the_interactive_command(spec) -> None:
    """The path a person actually takes, not only the parser."""

    from zippergen.provider_connections import connector_kinds_for_provider

    assert spec.name in connector_kinds_for_provider(spec.provider)


def test_an_undeclared_kind_has_no_spec_anywhere() -> None:
    from zippergen.connectors import connector_kind_spec

    assert connector_kind_spec("google-calendar") is None
    assert connector_kind_spec("") is None
    assert connector_kind_spec(None) is None


def test_no_module_writes_a_kind_grouping_out_by_hand() -> None:
    """Any literal set or tuple of connector-kind names is a second registry.

    Order-insensitive, unlike the pattern this replaces -- which searched for
    one spelling and let the same duplicate written the other way through.
    """

    import itertools
    import pathlib
    import re

    names = [spec.name for spec in CONNECTOR_KIND_SPECS]
    patterns = [
        re.compile(
            r"[\{\(\[]\s*"
            + r"\s*,\s*".join(f'"{re.escape(n)}"' for n in order)
            + r"\s*,?\s*[\}\)\]]"
        )
        for size in (2, 3)
        for order in itertools.permutations(names, size)
    ]
    source_root = pathlib.Path(__file__).resolve().parents[1] / "src" / "zippergen"
    offenders = sorted(
        path.name
        for path in source_root.rglob("*.py")
        if path.name != "connectors.py"
        and any(p.search(path.read_text()) for p in patterns)
    )
    assert not offenders, (
        "these modules group connector kinds by hand instead of reading "
        f"CONNECTOR_KIND_SPECS: {offenders}"
    )


def test_every_declared_kind_is_rendered_as_a_route() -> None:
    assert set(CONNECTOR_KINDS) <= set(CONNECTOR_ROUTE_KINDS)


def test_a_kind_no_provider_serves_is_refused_at_the_workflow() -> None:
    with pytest.raises(ValueError, match="google-calendar"):
        ConnectorRequirement(
            name="agenda",
            kind="google-calendar",
            access="read-only",
            participant="Assistant",
        )
