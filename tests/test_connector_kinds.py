"""Every declared connector kind must be servable and configurable.

A workflow may name any kind ``connectors.CONNECTOR_KINDS`` accepts. Accepting
one that no provider serves produces a workflow that validates, deploys, and
then cannot be configured -- the failure lands on the operator, far from the
line that caused it. These tests hold the declaration and the support together
so a kind can only be added once it is actually supported end to end.
"""

from zippergen.configuration_rendering import CONNECTOR_ROUTE_KINDS
from zippergen.connectors import CONNECTOR_KINDS, ConnectorRequirement
from zippergen.provider_connections import _CONNECTOR_KINDS
from zippergen.serve import _parse_cli_args

import pytest


def test_every_declared_kind_has_a_provider_that_serves_it() -> None:
    served = set().union(*_CONNECTOR_KINDS.values())
    assert served == set(CONNECTOR_KINDS)


def test_every_declared_kind_can_be_configured_from_the_command_line() -> None:
    for kind in CONNECTOR_KINDS:
        _, namespace = _parse_cli_args(
            ["connector", "configure", "records", "work", kind]
        )
        assert namespace.kind == kind


def test_every_declared_kind_is_offered_by_the_interactive_command() -> None:
    """The path a person actually takes, not only the parser.

    `zg connector configure` picks the kinds for a chosen provider. It used to
    write that mapping out by hand, so a new kind passed every test here and
    was still refused at the prompt.
    """

    from zippergen.provider_connections import (
        connector_kinds_for_provider,
        providers_serving_connectors,
    )

    offered: set[str] = set()
    for provider in providers_serving_connectors():
        offered |= set(connector_kinds_for_provider(provider))
    assert offered == set(CONNECTOR_KINDS)


def test_no_module_writes_the_provider_mapping_out_by_hand() -> None:
    """One statement of provider-to-connector compatibility, not four."""

    import pathlib
    import re

    source_root = pathlib.Path(__file__).resolve().parents[1] / "src" / "zippergen"
    pair = re.compile(r'"gmail",\s*"google-sheets"')
    offenders = [
        path.name
        for path in source_root.rglob("*.py")
        if path.name not in {"provider_connections.py", "connectors.py"}
        and pair.search(path.read_text())
    ]
    assert not offenders, (
        "these modules rebuild the provider mapping instead of asking "
        f"connector_kinds_for_provider: {offenders}"
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
