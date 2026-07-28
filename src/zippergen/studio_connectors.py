"""Studio connector provider, configuration, and assignment management."""

from __future__ import annotations

import time
from urllib import request

from zippergen.workspace import WorkspaceError

# This mixin uses Studio's rendering, selection, workflow-context, and human
# action discovery interface. Connector state and commands live here so the
# main shell owns orchestration rather than every domain implementation.
# pyright: reportAttributeAccessIssue=false, reportUnknownMemberType=false


class StudioConnectorsMixin:
    def _connector_requirements(self):
        from zippergen.connectors import connector_requirements_from_module

        current, _workflow, module = self._current_context()
        return current, connector_requirements_from_module(module)

    def _emit_connectors(self) -> None:
        providers = self.workspace.connector_provider_profiles()
        if providers:
            self._emit_columns(
                "Connector providers",
                ("Provider", "Status", "Detail"),
                [
                    (
                        name,
                        value.get("check_status", "not checked"),
                        value.get("check_detail", "—"),
                    )
                    for name, value in providers.items()
                ],
            )
        else:
            self._emit_table(
                "Connector providers",
                [
                    ("Status", "none configured", "warning"),
                    ("Next", "connector provider configure telegram", None),
                ],
            )

        configurations = self.workspace.connector_configurations()
        if configurations:
            self._emit_columns(
                "Connector configurations",
                ("Configuration", "Provider", "Resource", "Last check"),
                [
                    (
                        name,
                        value.get("provider")
                        or value.get("kind")
                        or "—",
                        value.get("chat_id")
                        or value.get("resource")
                        or "—",
                        (
                            f"{value.get('check_status', 'not checked')} — "
                            f"{value.get('check_detail', '')}"
                        ).rstrip(" —"),
                    )
                    for name, value in configurations.items()
                ],
            )
        else:
            self._emit_table(
                "Connector configurations",
                [
                    ("Status", "none", "warning"),
                    (
                        "Next",
                        "connector config create",
                        None,
                    ),
                ],
            )

        try:
            current, workflow, module = self._current_context()
        except SystemExit:
            self._emit_table(
                "Connector assignments",
                [
                    ("Workflow", "none selected", "warning"),
                    ("Assignments", "not available", None),
                ],
            )
            return

        human = self._human_action_lifelines(workflow, module)
        assignments = self.workspace.connector_assignment_profile(current)
        lifelines = assignments["lifelines"]
        actions = assignments["actions"]
        if not human:
            self._emit_table(
                "Human connector assignments",
                [
                    ("Workflow", current, None),
                    ("Status", "no human actions", None),
                ],
            )
        else:
            rows: list[tuple[object, ...]] = []
            for participant, action_names in human.items():
                for action_name in action_names:
                    target = f"{participant}.{action_name}"
                    explicit_action = actions.get(target)
                    participant_route = lifelines.get(participant)
                    effective = explicit_action or participant_route
                    source = (
                        "action override"
                        if explicit_action
                        else "participant"
                        if participant_route
                        else "local terminal"
                    )
                    configuration = configurations.get(effective or "")
                    provider = (
                        configuration.get("provider")
                        or configuration.get("kind")
                        if configuration
                        else "terminal"
                    )
                    rows.append(
                        (
                            participant,
                            action_name,
                            effective or "terminal",
                            provider,
                            source,
                        )
                    )
            self._emit_columns(
                "Human connector assignments",
                (
                    "Participant",
                    "Human action",
                    "Configuration",
                    "Provider",
                    "Source",
                ),
                rows,
            )
        self._emit_next("connector setup · connector assignments")

    def _check_connector_provider(self, name: str) -> bool:
        provider = name.casefold()
        profiles = self.workspace.connector_provider_profiles()
        profile = profiles.get(provider)
        if profile is None:
            raise SystemExit(
                f"Connector provider is not configured: {provider}."
            )
        if provider != "telegram":
            raise SystemExit(
                f"Live checks are not implemented for connector provider "
                f"{provider!r}."
            )
        token = self.workspace.connector_provider_secret(
            provider, "bot_token"
        )
        status = "unavailable"
        detail = ""
        try:
            if not token:
                raise ValueError("bot token is missing")
            from zippergen.telegram_notify import TelegramBotClient

            identity = TelegramBotClient(token, timeout=10).request(
                "getMe"
            ).get("result") or {}
            username = (
                f"@{identity.get('username')}"
                if isinstance(identity, dict) and identity.get("username")
                else "Telegram bot"
            )
            status = "available"
            detail = f"{username} authenticated"
        except Exception as exc:
            detail = str(exc)
        self.workspace.save_connector_provider_profile(
            provider,
            {
                **profile,
                "kind": provider,
                "check_status": status,
                "check_detail": detail,
                "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )
        emit = self._success if status == "available" else self._error
        emit(f"Connector provider {provider}: {detail}.")
        return status == "available"

    def _check_connector_configuration(self, name: str) -> bool:
        configurations = self.workspace.connector_configurations()
        configuration = configurations.get(name)
        if configuration is None:
            raise SystemExit(
                f"Connector configuration does not exist: {name}."
            )
        provider = str(
            configuration.get("provider")
            or configuration.get("kind")
            or ""
        )
        if provider != "telegram":
            raise SystemExit(
                f"Live checks are not implemented for connector provider "
                f"{provider!r}."
            )
        token = self.workspace.connector_provider_secret(
            provider, "bot_token"
        ) or self.workspace.connector_secret(name, "bot_token")
        chat_id = str(configuration.get("chat_id") or "")
        status = "failed"
        detail = ""
        try:
            if not token:
                raise ValueError("bot token is missing")
            if not chat_id:
                raise ValueError("chat id is missing")
            from zippergen.telegram_notify import TelegramBotClient

            client = TelegramBotClient(token, timeout=10)
            identity = client.request("getMe").get("result") or {}
            client.request("getChat", chat_id=chat_id)
            username = (
                f"@{identity.get('username')}"
                if isinstance(identity, dict) and identity.get("username")
                else "bot"
            )
            status = "available"
            detail = f"{username}; chat {chat_id} reachable"
        except Exception as exc:
            detail = str(exc)
        checked = {
            **configuration,
            "check_status": status,
            "check_detail": detail,
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        self.workspace.save_connector_configuration(name, checked)
        if status == "available":
            self._success(f"Connector {name}: {detail}.")
            return True
        self._error(f"Connector {name}: {detail}.")
        return False

    def _connector_configuration_name(self, requested: str) -> str:
        configurations = self.workspace.connector_configurations()
        name = {
            candidate.casefold(): candidate
            for candidate in configurations
        }.get(requested.casefold())
        if name is None:
            raise SystemExit(
                f"Connector configuration does not exist: {requested}. "
                f"Available: {', '.join(configurations) or 'none'}."
            )
        return name

    def _show_connector_configuration(self, requested: str) -> None:
        name = self._connector_configuration_name(requested)
        configuration = self.workspace.connector_configurations()[name]
        references = self.workspace.connector_configuration_references(name)
        self._emit_table(
            "Connector configuration",
            [
                ("Name", name, None),
                (
                    "Provider",
                    configuration.get("provider")
                    or configuration.get("kind")
                    or "unknown",
                    None,
                ),
                (
                    "Resource",
                    configuration.get("chat_id")
                    or configuration.get("resource")
                    or "—",
                    None,
                ),
                (
                    "Last check",
                    f"{configuration.get('check_status', 'not checked')} — "
                    f"{configuration.get('check_detail', '')}".rstrip(" —"),
                    None,
                ),
                (
                    "Used by",
                    ", ".join(
                        f"{workflow} ({kind} {target})"
                        for workflow, kind, target in references
                    )
                    if references
                    else "none",
                    None,
                ),
            ],
        )

    def _configure_connector_provider(self, provider: str) -> None:
        name = provider.casefold()
        if name != "telegram":
            raise SystemExit(
                "Telegram is the first supported human connector provider."
            )
        current = self.workspace.connector_provider_secret(
            name, "bot_token"
        )
        prompt = (
            "Telegram bot token [press Enter to keep stored token]: "
            if current
            else "Telegram bot token: "
        )
        token = self.secret_input(prompt).strip() or current
        if not token:
            raise SystemExit("Telegram bot token must not be empty.")
        self.workspace.save_connector_provider_secret(
            name, "bot_token", token
        )
        self.workspace.save_connector_provider_profile(
            name,
            {
                "kind": name,
                "check_status": "not checked",
                "check_detail": "credentials changed",
            },
        )
        self._success(
            "Configured Telegram provider; the bot token is private."
        )
        self._check_connector_provider(name)

    def _configure_connector_configuration(
        self,
        requested_name: str | None,
        *,
        edit: bool = False,
    ) -> str:
        configurations = self.workspace.connector_configurations()
        if edit:
            assert requested_name is not None
            name = self._connector_configuration_name(requested_name)
            existing = configurations[name]
        else:
            name = requested_name or "telegram-approvals"
            existing = configurations.get(name, {})
        providers = self.workspace.connector_provider_profiles()
        if not providers:
            raise SystemExit(
                "No connector provider is configured. Use "
                "'connector provider configure telegram' first."
            )
        current_provider = str(
            existing.get("provider")
            or existing.get("kind")
            or next(iter(providers))
        )
        if len(providers) == 1:
            provider = next(iter(providers))
        else:
            provider = str(
                self._select(
                    "Connector providers",
                    list(providers),
                    prompt="Select provider",
                )
            )
        current_chat = str(existing.get("chat_id") or "")
        chat_id = self.input(
            f"Telegram chat id [{current_chat}]: "
            if current_chat
            else "Telegram chat id: "
        ).strip() or current_chat
        if not chat_id:
            raise SystemExit("Telegram chat id must not be empty.")
        self.workspace.save_connector_configuration(
            name,
            {
                "provider": provider or current_provider,
                "kind": provider or current_provider,
                "chat_id": chat_id,
                "channel": f"telegram:{name}",
                "check_status": "not checked",
                "check_detail": "configuration changed",
            },
        )
        self._success(f"Saved connector configuration {name}.")
        self._check_connector_configuration(name)
        return name

    def _show_connector_assignments(self) -> None:
        current, workflow, module = self._current_context()
        self._emit_connectors()

    def _assign_connector(self, args: list[str]) -> None:
        if len(args) != 2:
            raise SystemExit(
                "Use connector assign PARTICIPANT_OR_ACTION CONFIGURATION."
            )
        entered_target, entered_configuration = args
        current, workflow, module = self._current_context()
        human = self._human_action_lifelines(workflow, module)
        action_targets = self._human_action_targets(workflow, module)
        participant = {
            name.casefold(): name for name in human
        }.get(entered_target.casefold())
        action_target = {
            name.casefold(): name for name in action_targets
        }.get(entered_target.casefold())
        if participant is None and action_target is None:
            available = ", ".join([*human, *action_targets]) or "none"
            raise SystemExit(
                f"{entered_target!r} is not a human participant or action. "
                f"Available targets: {available}."
            )
        configuration_name = self._connector_configuration_name(
            entered_configuration
        )
        configuration = self.workspace.connector_configurations()[
            configuration_name
        ]
        if (
            configuration.get("provider")
            or configuration.get("kind")
        ) != "telegram":
            raise SystemExit(
                f"{configuration_name} cannot deliver human actions."
            )
        status = configuration.get("check_status")
        if status in {"failed", "unavailable"}:
            raise SystemExit(
                f"{configuration_name} is unavailable. Run "
                f"'connector config check {configuration_name}' after fixing "
                "the provider or destination."
            )
        if status != "available":
            self._warning(
                f"{configuration_name} has not passed a live check. Use "
                f"'connector config check {configuration_name}'."
            )
        profile = self.workspace.connector_assignment_profile(current)
        lifelines = dict(profile["lifelines"])
        actions = dict(profile["actions"])
        target = action_target or participant
        assert target is not None
        if action_target:
            actions[action_target] = configuration_name
        else:
            lifelines[participant] = configuration_name  # type: ignore[index]
        self.workspace.save_connector_assignment_profile(
            current,
            lifelines=lifelines,
            actions=actions,
        )
        self._success(f"Assigned {configuration_name} to {target}.")
        self._emit_connectors()

    def _inherit_connector(self, args: list[str]) -> None:
        if len(args) != 1:
            raise SystemExit(
                "Use connector inherit PARTICIPANT_OR_ACTION."
            )
        current, workflow, module = self._current_context()
        human = self._human_action_lifelines(workflow, module)
        action_targets = self._human_action_targets(workflow, module)
        participant = {
            name.casefold(): name for name in human
        }.get(args[0].casefold())
        action_target = {
            name.casefold(): name for name in action_targets
        }.get(args[0].casefold())
        if participant is None and action_target is None:
            raise SystemExit(
                f"Unknown human participant or action: {args[0]}."
            )
        profile = self.workspace.connector_assignment_profile(current)
        lifelines = dict(profile["lifelines"])
        actions = dict(profile["actions"])
        if action_target:
            actions.pop(action_target, None)
            message = (
                f"{action_target} now inherits its participant route."
            )
        else:
            assert participant is not None
            lifelines.pop(participant, None)
            message = (
                f"{participant} now uses the local terminal unless an "
                "action override is assigned."
            )
        self.workspace.save_connector_assignment_profile(
            current,
            lifelines=lifelines,
            actions=actions,
        )
        self._success(message)
        self._emit_connectors()

    def manage_connectors(self, args: list[str]) -> None:
        if not args or args in (["list"], ["show"]):
            self._emit_connectors()
            return
        action, *rest = args
        action = action.casefold()
        if action == "setup":
            if rest:
                raise SystemExit("Use connector setup.")
            if "telegram" not in self.workspace.connector_provider_profiles():
                self._configure_connector_provider("telegram")
            name = self._configure_connector_configuration(None)
            current, workflow, module = self._current_context()
            human = self._human_action_lifelines(workflow, module)
            profile = self.workspace.connector_assignment_profile(current)
            lifelines = dict(profile["lifelines"])
            for participant in human:
                lifelines[participant] = name
            self.workspace.save_connector_assignment_profile(
                current,
                lifelines=lifelines,
                actions=profile["actions"],
            )
            self._success(
                f"Connector setup complete; {name} is assigned to "
                f"{len(human)} human participant(s)."
            )
            self._emit_connectors()
            return
        if action == "provider":
            subaction = rest[0].casefold() if rest else "list"
            values = rest[1:]
            if subaction in {"list", "show"} and not values:
                self._emit_connectors()
                return
            if subaction == "configure" and len(values) == 1:
                self._configure_connector_provider(values[0])
                self._emit_connectors()
                return
            if subaction == "check" and len(values) <= 1:
                names = (
                    list(self.workspace.connector_provider_profiles())
                    if not values or values[0] == "all"
                    else [values[0]]
                )
                failed = [
                    name for name in names
                    if not self._check_connector_provider(name)
                ]
                if failed:
                    raise SystemExit(
                        "Connector provider checks failed: "
                        + ", ".join(failed) + "."
                    )
                return
            if subaction == "remove" and len(values) == 1:
                try:
                    self.workspace.remove_connector_provider_profile(
                        values[0]
                    )
                except WorkspaceError as exc:
                    raise SystemExit(str(exc)) from exc
                self._success(
                    f"Removed connector provider: {values[0].casefold()}"
                )
                return
            raise SystemExit(
                "Use connector provider list, configure telegram, "
                "check [telegram|all], or remove telegram."
            )
        if action == "config":
            subaction = rest[0].casefold() if rest else "list"
            values = rest[1:]
            if subaction == "list" and not values:
                self._emit_connectors()
                return
            if subaction == "show" and len(values) <= 1:
                if not values or values[0] == "all":
                    self._emit_connectors()
                else:
                    self._show_connector_configuration(values[0])
                return
            if subaction == "create" and len(values) <= 1:
                self._configure_connector_configuration(
                    values[0] if values else None
                )
                return
            if subaction == "edit" and len(values) == 1:
                self._configure_connector_configuration(
                    values[0], edit=True
                )
                return
            if subaction == "check" and len(values) <= 1:
                names = (
                    list(self.workspace.connector_configurations())
                    if not values or values[0] == "all"
                    else [self._connector_configuration_name(values[0])]
                )
                failed = [
                    name for name in names
                    if not self._check_connector_configuration(name)
                ]
                if failed:
                    raise SystemExit(
                        "Connector configuration checks failed: "
                        + ", ".join(failed) + "."
                    )
                return
            if subaction == "rename" and len(values) == 2:
                self.workspace.rename_connector_configuration(
                    self._connector_configuration_name(values[0]),
                    values[1],
                )
                self._success(
                    f"Renamed connector configuration {values[0]} to "
                    f"{values[1]}."
                )
                return
            if subaction == "remove" and len(values) == 1:
                name = self._connector_configuration_name(values[0])
                self.workspace.remove_connector_configuration(name)
                self._success(f"Removed connector configuration: {name}")
                return
            raise SystemExit(
                "Use connector config list, create [NAME], edit NAME, "
                "check [NAME|all], rename OLD NEW, or remove NAME."
            )
        if action == "assignments":
            if not rest:
                self._show_connector_assignments()
                return
            if rest == ["check"]:
                current, workflow, module = self._current_context()
                self._check_workflow_connectors(
                    current, workflow, module
                )
                return
            raise SystemExit(
                "Use connector assignments or connector assignments check."
            )
        if action == "assign":
            self._assign_connector(rest)
            return
        if action == "inherit":
            self._inherit_connector(rest)
            return
        if action == "bind":
            if len(rest) != 2:
                raise SystemExit(
                    "Use connector bind REQUIREMENT CONFIGURATION."
                )
            requirement_name, configuration_name = rest
            current, requirements = self._connector_requirements()
            requirement = next(
                (
                    item
                    for item in requirements
                    if item.name.casefold() == requirement_name.casefold()
                ),
                None,
            )
            if requirement is None:
                declared = ", ".join(
                    item.name for item in requirements
                ) or "none"
                request_record = self.workspace.current_request()
                request_status = (
                    str(request_record.get("status") or "")
                    if request_record is not None
                    else ""
                )
                if request_status == "awaiting_review":
                    correction = "workflow implement --rerun"
                elif request_status == "prepared":
                    correction = "workflow implement"
                else:
                    correction = "workflow refine"
                self._emit_table(
                    "Connector binding blocked",
                    [
                        ("Workflow", current, None),
                        ("Requested", requirement_name, "warning"),
                        ("Configuration", configuration_name, None),
                        (
                            "Declared requirements",
                            declared,
                            "warning" if not requirements else None,
                        ),
                        (
                            "Reason",
                            (
                                "the selected workflow declares no logical "
                                "connector requirements"
                                if not requirements
                                else "the requested name is not among the "
                                "workflow's declared requirements"
                            ),
                            "warning",
                        ),
                    ],
                )
                self._emit_next(
                    f"workflow show full · {correction} · connector"
                )
                raise SystemExit(
                    f"Cannot bind {requirement_name}. The selected workflow "
                    "does not declare that connector requirement."
                )
            configurations = self.workspace.connector_configurations()
            configuration = configurations.get(configuration_name)
            if configuration is None:
                raise SystemExit(
                    f"Connector configuration does not exist: "
                    f"{configuration_name}."
                )
            if configuration.get("kind") != requirement.kind:
                raise SystemExit(
                    f"{requirement.name} requires {requirement.kind}, but "
                    f"{configuration_name} is {configuration.get('kind')}."
                )
            self.workspace.bind_connector(
                current,
                requirement.name,
                configuration_name,
            )
            self._success(
                f"Bound {requirement.name} to {configuration_name}."
            )
            self._emit_connectors()
            return
        raise SystemExit(
            "Use connector, connector setup, connector provider ..., "
            "connector config ..., connector assignments, connector assign, "
            "connector inherit, or connector bind."
        )
