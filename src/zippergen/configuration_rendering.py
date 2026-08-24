"""Render configuration reports without discovering or mutating state."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from zippergen.configuration_inventory import _provider
from zippergen.configuration_reporting import _selected_checks
from zippergen.rendering import TerminalRenderer


CONNECTOR_ROUTE_KINDS = (
    "human",
    "telegram",
    "gmail",
    "google-sheets",
    "google-calendar",
)


def _routing_status(renderer: TerminalRenderer, item: Mapping[str, object]) -> str:
    """Three states, because "configured" and "reached" are different news.

    A command that never contacts a provider cannot honestly claim a route
    works; it can only say nothing contradicts it. `zg config` is offline and
    `zg check` is live, so the depth of the answer belongs on the row rather
    than in the reader's memory of which command they typed.
    """

    if not item.get("available"):
        return renderer.status_mark("error")
    return renderer.status_mark("success" if item.get("verified") else "info")


def _render_connector_routing(
    renderer: TerminalRenderer,
    report: Mapping[str, object],
) -> None:
    """Both connector views ask the same question, so they share one answer."""

    connectors = report.get("connectors") or {}
    raw = connectors.get("configurations") if isinstance(connectors, dict) else []
    resources = {
        str(item.get("name")): str(
            item.get("chat_id")
            or item.get("spreadsheet_id")
            or item.get("query")
            or item.get("account")
            or "-"
        )
        for item in raw or []
        if isinstance(item, dict)
    }

    def resource(item: Mapping[str, object]) -> str:
        # The full value is in the Configurations table above, so this one
        # only has to be recognisable; a long id would squeeze out the
        # columns that carry the answer.
        text = resources.get(str(item.get("configuration")), "-")
        return text if len(text) <= 22 else text[:21] + "\u2026"

    _render_effective_routing(
        renderer,
        report,
        CONNECTOR_ROUTE_KINDS,
        subject="Slot",
        resolved_header="Resource",
        resolved=resource,
        empty="Nothing reaches outside this workflow.",
    )


def _render_effective_routing(
    renderer: TerminalRenderer,
    report: Mapping[str, object],
    kinds: tuple[str, ...],
    *,
    subject: str,
    resolved_header: str,
    resolved: Callable[[Mapping[str, object]], object],
    empty: str,
) -> None:
    """Answer "what will this use, and where do I change it?" for one family.

    The participant is printed once per group rather than on every row, so the
    shape of the answer is visible before any of it is read.
    """

    routes = report.get("effective_routing") or []
    rows: list[tuple[object, ...]] = []
    previous = None
    for item in routes if isinstance(routes, list) else []:
        if not isinstance(item, dict) or item.get("kind") not in kinds:
            continue
        participant = item.get("participant")
        rows.append((
            _routing_status(renderer, item),
            "" if participant == previous else participant,
            item.get("action"),
            item.get("configuration"),
            resolved(item),
            item.get("source"),
        ))
        previous = participant
    _render_columns_or_empty(
        renderer,
        "Effective routing",
        ("", "Participant", subject, "Configuration", resolved_header, "From"),
        rows,
        empty=empty,
    )


def _nested_assignment_rows(
    assignments: dict[str, object],
) -> list[tuple[object, object, object]]:
    """Render participant routes with exact-action overrides directly below."""

    rows: list[tuple[object, object, object]] = []
    default = assignments.get("default")
    if default:
        rows.append(("default", default, "default"))
    raw_lifelines = assignments.get("lifelines")
    raw_actions = assignments.get("actions")
    lifelines = (
        {
            str(target): str(configuration)
            for target, configuration in raw_lifelines.items()
        }
        if isinstance(raw_lifelines, Mapping)
        else {}
    )
    actions = (
        {
            str(target): str(configuration)
            for target, configuration in raw_actions.items()
        }
        if isinstance(raw_actions, Mapping)
        else {}
    )
    participants = sorted(
        {*lifelines, *(target.partition(".")[0] for target in actions)}
    )
    for participant in participants:
        rows.append(
            (
                participant,
                lifelines.get(participant) or "inherits default",
                "participant" if participant in lifelines else "inherited",
            )
        )
        for target, configuration in sorted(actions.items()):
            owner, separator, action = target.partition(".")
            if separator and owner == participant:
                rows.append((f"  {action}", configuration, "action override"))
    return rows


def _configuration_display(row: dict) -> str:
    """Render one answer without printing a secret."""

    if row.get("secret"):
        return "(stored privately)" if row.get("answered") else "(not set)"
    if not row.get("answered"):
        default = row.get("default")
        return (
            f"(not set; default {default})"
            if default is not None
            else "(not set)"
        )
    return str(row.get("value"))


def _render_project_configuration(renderer, report: dict) -> None:
    """Show the project's answers to its workflow's declared questions.

    They are stored in the project manifest beside every other choice, so the
    command that shows a project's configuration shows them too.
    """

    rows = report.get("configuration") or []
    if not isinstance(rows, list):
        return
    renderer.framed_section("Configuration")
    _render_columns_or_empty(
        renderer,
        "Answers",
        ("Field", "Value", "Asks"),
        [
            (
                row.get("name"),
                _configuration_display(row),
                row.get("prompt") or "",
            )
            for row in rows
            if isinstance(row, dict)
        ],
        empty="This workflow declares no configuration.",
    )


def _render_columns_or_empty(
    renderer: TerminalRenderer,
    title: str,
    headers: tuple[str, ...],
    rows: Sequence[tuple[object, ...]],
    *,
    empty: str,
) -> None:
    """Render a configuration subsection without an empty table shell."""

    if rows:
        renderer.columns(title, headers, list(rows))
    else:
        renderer.empty(title, empty)


def _idle_release_display(item: Mapping[str, object]) -> str:
    value = item.get("idle_timeout")
    provider = _provider(str(item.get("spec") or ""))
    if provider != "local":
        return "not applicable"
    if value is None or str(value).strip() == "":
        return "not set"
    seconds = float(str(value))
    return "after each call" if seconds == 0 else f"after {seconds:g} s"


def _effective_model_display(item: Mapping[str, object]) -> str:
    settings = item.get("settings")
    values = dict(settings) if isinstance(settings, Mapping) else {}
    temperature = values.get("temperature", item.get("temperature"))
    parts = [
        f"T={float(str(temperature)):g}"
        if temperature is not None
        else "T=provider"
    ]
    for name, label in (
        ("max_tokens", "max"),
        ("timeout", "timeout"),
        ("idle_timeout", "idle"),
    ):
        value = values.get(name)
        if value is not None:
            suffix = "s" if name in {"timeout", "idle_timeout"} else ""
            parts.append(f"{label}={float(str(value)):g}{suffix}")
    return f"{item.get('effective')} · " + " · ".join(parts)


def render_configuration(
    report: dict[str, object],
    renderer: TerminalRenderer,
    *,
    show_checks: bool = False,
) -> None:
    project = report["project"]
    assert isinstance(project, dict)
    renderer.framed_section("Project")
    renderer.table(
        "Details",
        [
            ("Project", project.get("name"), None),
            ("Root", project.get("root"), None),
            ("Workflow", project.get("workflow") or "not resolved", None),
            ("Specification", project.get("specification"), None),
            ("Manifest", project.get("manifest"), None),
        ],
    )
    _render_project_configuration(renderer, report)
    renderer.framed_section("Providers")
    providers = report.get("providers") or {}
    assert isinstance(providers, dict)
    _render_columns_or_empty(
        renderer,
        "Connections",
        ("Name", "Kind", "Site endpoint"),
        [
            (
                item.get("name"),
                item.get("kind"),
                item.get("base_url") or "provider default",
            )
            for item in providers.get("connections") or []
            if isinstance(item, dict)
        ],
        empty="No connections.",
    )
    renderer.framed_section("Models")
    models = report["models"]
    assert isinstance(models, dict)
    configurations = models.get("configurations") or []
    model_configuration_rows = [
        (
            item.get("name"),
            item.get("connection") or "-",
            item.get("model") or "-",
            item.get("temperature") if item.get("temperature") is not None else "default",
            item.get("max_tokens") if item.get("max_tokens") is not None else "default",
            item.get("timeout") if item.get("timeout") is not None else "default",
            _idle_release_display(item),
            item.get("source"),
        )
        for item in configurations
        if isinstance(item, dict)
    ]
    _render_columns_or_empty(
        renderer,
        "Configurations",
        (
            "Name",
            "Connection",
            "Model",
            "Temperature",
            "Max tokens",
            "Timeout",
            "Idle release",
            "Source",
        ),
        model_configuration_rows,
        empty="No configurations.",
    )
    assignments = models.get("assignments") or {}
    assert isinstance(assignments, dict)
    _render_columns_or_empty(
        renderer,
        "Assignments",
        ("Target", "Configuration", "Scope"),
        _nested_assignment_rows(assignments),
        empty="No assignments.",
    )
    _render_effective_routing(
        renderer,
        report,
        ("model",),
        subject="Action",
        resolved_header="Resolves to",
        resolved=_effective_model_display,
        empty="No participant calls a model.",
    )
    renderer.framed_section("Assistants")
    _render_assistant_tables(report, renderer, compact_titles=True)
    renderer.framed_section("Connectors")
    connectors = report["connectors"]
    assert isinstance(connectors, dict)
    connector_configuration_rows = [
        (
            item.get("name"),
            item.get("kind") or item.get("provider"),
            item.get("connection") or "-",
            item.get("chat_id")
            or item.get("spreadsheet_id")
            or item.get("query")
            or "-",
        )
        for item in connectors.get("configurations") or []
        if isinstance(item, dict)
    ]
    _render_columns_or_empty(
        renderer,
        "Configurations",
        ("Name", "Kind", "Connection", "Resource"),
        connector_configuration_rows,
        empty="No configurations.",
    )
    _render_columns_or_empty(
        renderer,
        "Slots",
        ("Target", "What it is", "Configuration"),
        [
            (item["target"], item["meaning"], item["configuration"])
            for item in (connectors.get("slots") or [])
            if isinstance(item, dict)
        ],
        empty="This workflow has no connector slots.",
    )
    _render_connector_routing(renderer, report)
    renderer.framed_section("Site")
    renderer.table(
        "Private state",
        [
            (
                "Location",
                report.get("site_root") or "not available",
                None,
            )
        ],
    )
    raw_site_facts = report.get("site_facts") or []
    site_facts = raw_site_facts if isinstance(raw_site_facts, list) else []
    _render_columns_or_empty(
        renderer,
        "Local requirements",
        ("Status", "Kind", "Requirement"),
        [
            (
                renderer.status_mark(
                    "success" if item.get("available") else "error"
                ),
                item.get("kind"),
                item.get("name"),
            )
            for item in site_facts
            if isinstance(item, dict)
        ],
        empty="No local credentials or tools are required.",
    )
    if not show_checks:
        return
    raw_checks = report.get("checks") or []
    checks = raw_checks if isinstance(raw_checks, list) else []
    renderer.framed_section("Readiness")
    _render_columns_or_empty(
        renderer,
        "Checks",
        ("Status", "Check", "Detail"),
        [
            (
                renderer.status_mark(
                    "success" if item.get("status") == "ok" else (
                        "warning" if item.get("status") == "warn" else "error"
                    )
                ),
                item.get("name"),
                item.get("detail"),
            )
            for item in checks
            if isinstance(item, dict)
        ],
        empty="No checks.",
    )


def _render_next_commands(renderer: TerminalRenderer, checks: list) -> None:
    """List the commands that would satisfy what is not ready.

    Reporting a hole without naming the command that fills it leaves a reader
    to reconstruct a dozen setup commands, and their order, from the manual.
    Only unsatisfied checks contribute, and each command is listed once.
    """

    commands: list[str] = []
    for item in checks:
        if not isinstance(item, dict) or item.get("status") == "ok":
            continue
        fix = str(item.get("fix") or "").strip()
        if fix and fix not in commands:
            commands.append(fix)
    if not commands:
        return
    renderer.emit("")
    renderer.emit("To finish, run:")
    for command in commands:
        renderer.emit(f"  {command}")


def render_readiness(
    report: dict[str, object],
    renderer: TerminalRenderer,
) -> None:
    """Render one live readiness view grouped by participant and dependency."""

    renderer.framed_section("Project readiness")
    project = report.get("project") or {}
    assert isinstance(project, dict)
    renderer.table(
        "Workflow",
        [
            ("Project", project.get("name"), None),
            ("Workflow", project.get("workflow") or "not resolved", None),
            (
                "Overall",
                "ready" if report.get("valid") else "not ready",
                "success" if report.get("valid") else "error",
            ),
        ],
    )
    raw_routes = report.get("effective_routing") or []
    routes = raw_routes if isinstance(raw_routes, list) else []
    route_rows = [item for item in routes if isinstance(item, dict)]
    previous = None
    rows: list[tuple[object, ...]] = []
    for item in route_rows:
        participant = item.get("participant")
        rows.append((
            _routing_status(renderer, item),
            "" if participant == previous else participant,
            item.get("action"),
            item.get("kind"),
            item.get("configuration"),
            item.get("source"),
        ))
        previous = participant
    _render_columns_or_empty(
        renderer,
        "Effective routing",
        ("", "Participant", "Action", "Kind", "Configuration", "From"),
        rows,
        empty="No model, assistant, human, or connector routes.",
    )
    raw_checks = report.get("checks") or []
    checks = raw_checks if isinstance(raw_checks, list) else []
    categories = (
        (
            "Structure and assignments",
            lambda name: "live " not in name
            and "credential" not in name
            and " CLI " not in f" {name} ",
        ),
        (
            "Credentials and local tools",
            lambda name: "credential" in name or " CLI " in f" {name} ",
        ),
        ("Live providers", lambda name: name.startswith("live ")),
    )
    for title, selected in categories:
        rows = []
        for item in checks:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "")
            if not selected(name):
                continue
            rows.append(
                (
                    renderer.status_mark(
                        "success" if item.get("status") == "ok" else (
                            "warning" if item.get("status") == "warn" else "error"
                        )
                    ),
                    name,
                    item.get("detail"),
                )
            )
        _render_columns_or_empty(
            renderer,
            title,
            ("Status", "Check", "Detail"),
            rows,
            empty="No checks.",
        )
    _render_next_commands(renderer, checks)


def render_model_configuration(
    report: dict[str, object],
    renderer: TerminalRenderer,
    *,
    show_checks: bool = True,
) -> None:
    """Render only the project's model configurations and effective routing."""

    models = report["models"]
    assert isinstance(models, dict)
    renderer.framed_section("Models")
    configuration_rows = [
        (
            item.get("name"),
            item.get("connection") or "-",
            item.get("model") or "-",
            item.get("temperature") if item.get("temperature") is not None else "default",
            item.get("max_tokens") if item.get("max_tokens") is not None else "default",
            item.get("timeout") if item.get("timeout") is not None else "default",
            _idle_release_display(item),
            item.get("source"),
        )
        for item in models.get("configurations") or []
        if isinstance(item, dict)
    ]
    _render_columns_or_empty(
        renderer,
        "Configurations",
        (
            "Name",
            "Connection",
            "Model",
            "Temperature",
            "Max tokens",
            "Timeout",
            "Idle release",
            "Source",
        ),
        configuration_rows,
        empty="No configurations.",
    )
    assignments = models.get("assignments") or {}
    assert isinstance(assignments, dict)
    _render_columns_or_empty(
        renderer,
        "Assignments",
        ("Target", "Configuration", "Scope"),
        _nested_assignment_rows(assignments),
        empty="No assignments.",
    )
    _render_effective_routing(
        renderer,
        report,
        ("model",),
        subject="Action",
        resolved_header="Resolves to",
        resolved=_effective_model_display,
        empty="No participant calls a model.",
    )
    if show_checks:
        _render_selected_checks(report, renderer, "model")


def _render_assistant_tables(
    report: dict[str, object],
    renderer: TerminalRenderer,
    *,
    compact_titles: bool = False,
) -> None:
    assistants = report["assistants"]
    assert isinstance(assistants, dict)
    configuration_rows = [
        (item.get("name"), item.get("backend"))
        for item in assistants.get("configurations") or []
        if isinstance(item, dict)
    ]
    _render_columns_or_empty(
        renderer,
        "Configurations" if compact_titles else "Assistant configurations",
        ("Name", "Backend"),
        configuration_rows,
        empty="No configurations.",
    )
    assignments = assistants.get("assignments") or {}
    assert isinstance(assignments, dict)
    _render_columns_or_empty(
        renderer,
        "Assignments" if compact_titles else "Assistant assignments",
        ("Target", "Configuration", "Scope"),
        _nested_assignment_rows(assignments),
        empty="No assignments.",
    )
    _render_effective_routing(
        renderer,
        report,
        ("assistant",),
        subject="Action",
        resolved_header="Backend",
        resolved=lambda item: item.get("effective"),
        empty="No participant runs an assistant.",
    )
    # Access, tools and shell are declared on the `@assistant` action, not
    # configured, so they answer "what may it do" rather than "what will it
    # use". Kept out of the routing table, where they sat at the right edge
    # and read as more routing.
    _render_columns_or_empty(
        renderer,
        "Permissions" if compact_titles else "Assistant permissions",
        ("Target", "Access", "Tools", "Shell"),
        [
            (
                item.get("target"),
                item.get("access"),
                item.get("external_tools"),
                item.get("shell"),
            )
            for item in assistants.get("resolved") or []
            if isinstance(item, dict)
        ],
        empty="No assistant actions.",
    )


def render_assistant_configuration(
    report: dict[str, object],
    renderer: TerminalRenderer,
    *,
    show_checks: bool = True,
) -> None:
    """Render coding-assistant configurations and effective routing."""

    renderer.framed_section("Assistants")
    _render_assistant_tables(report, renderer, compact_titles=True)
    if show_checks:
        _render_selected_checks(report, renderer, "assistant")


def render_connector_configuration(
    report: dict[str, object],
    renderer: TerminalRenderer,
    *,
    show_checks: bool = True,
) -> None:
    """Render only the project's connector configurations and routing."""

    connectors = report["connectors"]
    assert isinstance(connectors, dict)
    renderer.framed_section("Connectors")
    configuration_rows = [
        (
            item.get("name"),
            item.get("kind") or item.get("provider"),
            item.get("connection") or "-",
            item.get("chat_id")
            or item.get("spreadsheet_id")
            or item.get("query")
            or "-",
        )
        for item in connectors.get("configurations") or []
        if isinstance(item, dict)
    ]
    _render_columns_or_empty(
        renderer,
        "Configurations",
        ("Name", "Kind", "Connection", "Resource"),
        configuration_rows,
        empty="No configurations.",
    )
    _render_columns_or_empty(
        renderer,
        "Slots",
        ("Target", "What it is", "Configuration"),
        [
            (item["target"], item["meaning"], item["configuration"])
            for item in (connectors.get("slots") or [])
            if isinstance(item, dict)
        ],
        empty="This workflow has no connector slots.",
    )
    _render_connector_routing(renderer, report)
    if show_checks:
        _render_selected_checks(report, renderer, "connector")


def render_provider_configuration(
    report: dict[str, object],
    renderer: TerminalRenderer,
    *,
    show_checks: bool = True,
) -> None:
    """Render named provider connections and their local readiness."""

    renderer.framed_section("Providers")
    providers = report.get("providers") or {}
    assert isinstance(providers, dict)
    _render_columns_or_empty(
        renderer,
        "Connections",
        ("Name", "Kind", "Site endpoint"),
        [
            (
                item.get("name"),
                item.get("kind"),
                item.get("base_url") or "provider default",
            )
            for item in providers.get("connections") or []
            if isinstance(item, dict)
        ],
        empty="No connections.",
    )
    if show_checks:
        _render_selected_checks(report, renderer, "provider")


def _render_selected_checks(
    report: dict[str, object],
    renderer: TerminalRenderer,
    scope: str,
) -> None:
    checks = _selected_checks(report, scope)
    _render_columns_or_empty(
        renderer,
        "Checks",
        ("Status", "Check", "Detail"),
        [
            (
                renderer.status_mark(
                    "success" if item.get("status") == "ok" else (
                        "warning" if item.get("status") == "warn" else "error"
                    )
                ),
                item.get("name"),
                item.get("detail"),
            )
            for item in checks
        ],
        empty="No checks.",
    )
