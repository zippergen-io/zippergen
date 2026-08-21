"""Public facade for inspecting and managing project configuration.

Discovery, checking, rendering, and mutation live in separate modules so
each layer can be read and tested without pulling the others into view.
"""

from zippergen.configuration_inventory import _model_targets
from zippergen.configuration_mutations import (
    AMBIGUOUS_TARGET,
    CONNECTOR_REQUIREMENT,
    HUMAN_ACTION,
    assign_assistant,
    assign_connector,
    assign_model,
    assistant_target_problem,
    configure_assistant,
    configure_model,
    connector_target_kinds,
    connector_target_problem,
    project_google_scopes,
)
from zippergen.configuration_rendering import (
    _routing_status,
    render_assistant_configuration,
    render_configuration,
    render_connector_configuration,
    render_model_configuration,
    render_provider_configuration,
    render_readiness,
)
from zippergen.configuration_reporting import (
    configuration_report,
    configuration_scope_valid,
)

__all__ = [
    "AMBIGUOUS_TARGET",
    "CONNECTOR_REQUIREMENT",
    "HUMAN_ACTION",
    "assign_assistant",
    "assign_connector",
    "assign_model",
    "assistant_target_problem",
    "configuration_report",
    "configuration_scope_valid",
    "configure_assistant",
    "configure_model",
    "connector_target_kinds",
    "connector_target_problem",
    "project_google_scopes",
    "render_assistant_configuration",
    "render_configuration",
    "render_connector_configuration",
    "render_model_configuration",
    "render_provider_configuration",
    "render_readiness",
]
