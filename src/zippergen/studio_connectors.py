"""Studio connector provider, configuration, and assignment management."""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
import time
from pathlib import Path

from zippergen.workspace import WorkspaceError

# This mixin uses Studio's rendering, selection, workflow-context, and human
# action discovery interface. Connector state and commands live here so the
# main shell owns orchestration rather than every domain implementation.
# pyright: reportAttributeAccessIssue=false, reportUnknownMemberType=false


class StudioConnectorsMixin:
    @staticmethod
    def _read_google_client_json(path: Path) -> str:
        from zippergen.google_auth import (
            GoogleConnectorError,
            normalize_google_client_json,
        )

        expanded = path.expanduser().resolve()
        if not expanded.is_file():
            raise SystemExit(
                f"Google OAuth desktop client JSON does not exist: {expanded}"
            )
        try:
            value = expanded.read_text()
        except OSError as exc:
            raise SystemExit(
                f"Could not read Google OAuth desktop client JSON: {exc}"
            ) from exc
        try:
            return normalize_google_client_json(value)
        except GoogleConnectorError as exc:
            raise SystemExit(str(exc)) from exc

    def _upload_google_client_json(self) -> str:
        upload_directory = self.workspace.directory / "uploads"
        upload_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        upload_directory.chmod(0o700)
        upload_path = upload_directory / (
            f"google-oauth-client-{time.time_ns()}.json"
        )
        local_path = self.input(
            "Absolute JSON path on your local computer "
            "[/local/path/downloaded-client.json]: "
        ).strip() or "/local/path/downloaded-client.json"
        ssh_host = self.input(
            "SSH server or alias [AI_SERVER]: "
        ).strip() or "AI_SERVER"
        self._google_ssh_host_hint = ssh_host
        command = (
            f"scp {shlex.quote(local_path)} "
            f"{shlex.quote(f'{ssh_host}:{upload_path}')}"
        )
        self._emit_table(
            "Private Google client upload",
            [
                ("Run on", "your local computer", None),
                ("Command", command, None),
                (
                    "Destination",
                    "private temporary Studio storage",
                    "success",
                ),
            ],
        )
        confirmation = self.input(
            "Press Enter after the upload, or type 'cancel': "
        ).strip()
        if confirmation.casefold() == "cancel":
            raise SystemExit("Google provider configuration cancelled.")
        if not upload_path.is_file():
            raise SystemExit(
                "The uploaded Google OAuth client was not found. Run the "
                "displayed scp command, then try again."
            )
        try:
            upload_path.chmod(0o600)
            return self._read_google_client_json(upload_path)
        finally:
            upload_path.unlink(missing_ok=True)

    def _collect_google_client_json(self, profile: dict[str, str]) -> str:
        stored = self.workspace.connector_provider_secret(
            "google", "oauth_client_json"
        )
        choices = []
        if stored:
            choices.append("Use the client already stored privately")
        choices.extend(
            [
                "Use a file already on this computer",
                "Upload from another computer",
            ]
        )
        selected = self._select(
            "Google OAuth client",
            choices,
            prompt="Select client source",
        )
        assert isinstance(selected, str)
        if selected == "Use the client already stored privately":
            assert stored is not None
            return stored
        if selected == "Upload from another computer":
            return self._upload_google_client_json()

        legacy_path = str(profile.get("credentials_file") or "")
        entered = self.input(
            (
                f"Google OAuth desktop client JSON [{legacy_path}]: "
                if legacy_path
                else "Google OAuth desktop client JSON path: "
            )
        ).strip()
        selected_path = entered or legacy_path
        if not selected_path:
            raise SystemExit(
                "Select the OAuth Desktop app JSON downloaded from Google "
                "Cloud."
            )
        return self._read_google_client_json(Path(selected_path))

    def _needs_remote_google_browser(self) -> bool:
        """Recognize an interactive Studio session on an SSH-only host."""

        if not getattr(self, "_prompt_toolkit_enabled", False):
            return False
        if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
            return True
        return (
            sys.platform.startswith("linux")
            and not os.environ.get("DISPLAY")
            and not os.environ.get("WAYLAND_DISPLAY")
        )

    def _authorize_google_provider(
        self,
        client_json: str,
        *,
        scopes: tuple[str, ...],
        service_names: str,
        profile: dict[str, str],
    ) -> tuple[str, str | None]:
        from zippergen.google_auth import (
            authorize_google_client,
            available_google_callback_port,
        )

        if not self._needs_remote_google_browser():
            self._info(
                f"A browser window will ask you to authorize {service_names}. "
                "Studio stores the resulting token privately."
            )
            return authorize_google_client(client_json, scopes=scopes), None

        default_host = str(
            getattr(self, "_google_ssh_host_hint", "")
            or profile.get("authorization_ssh_host")
            or "AI_SERVER"
        )
        entered_host = self.input(
            f"SSH server or alias used from your browser computer "
            f"[{default_host}]: "
        ).strip()
        ssh_host = entered_host or default_host
        port = available_google_callback_port()
        forward = f"{port}:127.0.0.1:{port}"
        quoted_host = shlex.quote(ssh_host)
        self._emit_table(
            "Remote Google authorization",
            [
                ("Browser", "your local computer", None),
                (
                    "Check",
                    f"ssh -O check {quoted_host}",
                    None,
                ),
                (
                    "Forward",
                    f"ssh -O forward -L {forward} {quoted_host}",
                    None,
                ),
                (
                    "Without a master",
                    f"ssh -N -L {forward} {quoted_host}",
                    None,
                ),
                (
                    "Then",
                    "return here; Studio prints the Google URL to open locally",
                    None,
                ),
            ],
        )
        confirmation = self.input(
            "Press Enter after the SSH forwarding is ready, or type 'cancel': "
        ).strip()
        if confirmation.casefold() == "cancel":
            raise SystemExit("Google provider configuration cancelled.")
        self._info(
            f"Open the authorization URL below in your local browser. "
            f"The callback uses forwarded port {port}."
        )
        authorized = authorize_google_client(
            client_json,
            scopes=scopes,
            open_browser=False,
            port=port,
        )
        self._emit_table(
            "Remote authorization cleanup",
            [
                (
                    "Multiplexed tunnel",
                    f"ssh -O cancel -L {forward} {quoted_host}",
                    None,
                ),
                (
                    "Standalone tunnel",
                    "close the terminal running ssh -N -L",
                    None,
                ),
            ],
        )
        return authorized, ssh_host

    @staticmethod
    def _google_profile_scopes(profile) -> tuple[str, ...]:
        raw = profile.get("scopes") if profile else None
        if isinstance(raw, (tuple, list)):
            return tuple(str(value) for value in raw)
        if isinstance(raw, str) and raw:
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw.split(",")
            if isinstance(value, list):
                return tuple(str(item) for item in value)
        return ()

    @staticmethod
    def _google_scopes_for_requirements(
        requirements,
    ) -> tuple[str, ...]:
        from zippergen.google_auth import google_scopes_for_access

        return google_scopes_for_access(
            (str(kind), str(access))
            for kind, access in requirements
        )

    @staticmethod
    def _google_scopes_cover(
        configured,
        required,
    ) -> bool:
        from zippergen.google_auth import google_scopes_cover

        return google_scopes_cover(configured, required)

    def _google_requirement_pairs(
        self,
        *,
        kinds: tuple[str, ...] | None = None,
    ) -> tuple[tuple[str, str], ...]:
        try:
            _current, requirements = self._connector_requirements()
        except SystemExit:
            requirements = ()
        selected = {
            str(kind)
            for kind in kinds
        } if kinds is not None else {"google-sheets", "gmail"}
        return tuple(
            (requirement.kind, requirement.access)
            for requirement in requirements
            if requirement.kind in selected
        )

    @staticmethod
    def _google_spreadsheet_id(value: str) -> str:
        text = value.strip()
        match = re.search(r"/spreadsheets/d/([A-Za-z0-9_-]+)", text)
        return match.group(1) if match else text

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
                    ("Next", "connector setup", None),
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
        from zippergen.connectors import connector_requirements_from_module

        requirements = connector_requirements_from_module(module)
        bindings = self.workspace.connector_binding_profile(current)
        if requirements:
            self._emit_columns(
                "Service connector bindings",
                (
                    "Requirement",
                    "Participant",
                    "Kind",
                    "Access",
                    "Configuration",
                ),
                [
                    (
                        requirement.name,
                        requirement.participant,
                        requirement.kind,
                        requirement.access,
                        bindings.get(requirement.name) or "not bound",
                    )
                    for requirement in requirements
                ],
            )
        else:
            self._emit_table(
                "Service connector bindings",
                [
                    ("Status", "no service connector requirements", None),
                ],
            )
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
        if provider not in {"telegram", "google"}:
            raise SystemExit(
                f"Live checks are not implemented for connector provider "
                f"{provider!r}."
            )
        status = "unavailable"
        detail = ""
        try:
            if provider == "telegram":
                token = self.workspace.connector_provider_secret(
                    provider, "bot_token"
                )
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
                detail = f"{username} authenticated"
            else:
                authorized_user = self.workspace.connector_provider_secret(
                    provider, "authorized_user_json"
                )
                if not authorized_user:
                    raise ValueError("Google authorization is missing")
                from zippergen.google_auth import (
                    check_google_authorization,
                )

                scopes = self._google_profile_scopes(profile)
                refreshed = check_google_authorization(
                    authorized_user,
                    scopes=scopes or self._google_scopes_for_requirements(
                        (("google-sheets", "read-write"),)
                    ),
                )
                self.workspace.save_connector_provider_secret(
                    provider, "authorized_user_json", refreshed
                )
                detail = (
                    f"Google authorization refreshed for "
                    f"{len(scopes) or 1} service scope(s)"
                )
            status = "available"
        except Exception as exc:
            detail = str(exc)
        self.workspace.save_connector_provider_profile(
            provider,
            {
                **profile,
                "kind": profile.get("kind") or provider,
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
        provider = str(configuration.get("provider") or "")
        kind = str(configuration.get("kind") or "")
        if provider not in {"telegram", "google"}:
            raise SystemExit(
                f"Live checks are not implemented for connector provider "
                f"{provider!r}."
            )
        status = "failed"
        detail = ""
        try:
            if provider == "telegram":
                token = self.workspace.connector_provider_secret(
                    provider, "bot_token"
                ) or self.workspace.connector_secret(name, "bot_token")
                chat_id = str(configuration.get("chat_id") or "")
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
                detail = f"{username}; chat {chat_id} reachable"
            elif kind == "google-sheets":
                authorized_user = self.workspace.connector_provider_secret(
                    provider, "authorized_user_json"
                )
                if not authorized_user:
                    raise ValueError("Google authorization is missing")
                from zippergen.google_sheets import GoogleSheetsTable

                table = GoogleSheetsTable(
                    requirement=name,
                    spreadsheet_id=str(
                        configuration.get("spreadsheet_id") or ""
                    ),
                    tab=str(configuration.get("tab") or ""),
                    credential_json=authorized_user,
                    access="read-only",
                )
                info = table.inspect()
                detail = f"{info['title']} · tab {info['tab']}"
            elif kind == "gmail":
                authorized_user = self.workspace.connector_provider_secret(
                    provider, "authorized_user_json"
                )
                if not authorized_user:
                    raise ValueError("Google authorization is missing")
                from zippergen.google_gmail import GmailMailbox

                mailbox = GmailMailbox(
                    requirement=name,
                    account=str(configuration.get("account") or "me"),
                    query=str(
                        configuration.get("query")
                        or "is:unread in:inbox"
                    ),
                    credential_json=authorized_user,
                    access="read-only",
                )
                info = mailbox.inspect()
                detail = (
                    f"{info['email']} reachable · query "
                    f"{mailbox.query!r}"
                )
            else:
                raise ValueError(
                    f"provider google does not support connector kind {kind!r}"
                )
            status = "available"
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
                    or (
                        f"{configuration.get('spreadsheet_id')} · "
                        f"tab {configuration.get('tab')}"
                        if configuration.get("spreadsheet_id")
                        else None
                    )
                    or (
                        f"{configuration.get('account', 'me')} · "
                        f"query {configuration.get('query')}"
                        if configuration.get("kind") == "gmail"
                        else None
                    )
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

    def _configure_connector_provider(
        self,
        provider: str,
        *,
        google_requirements: tuple[tuple[str, str], ...] | None = None,
        preserve_google_scopes: bool = False,
    ) -> None:
        name = provider.casefold()
        scopes: tuple[str, ...] = ()
        authorization_ssh_host: str | None = None
        if name not in {"telegram", "google"}:
            raise SystemExit(
                "Supported connector providers are telegram and google."
            )
        if name == "telegram":
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
            detail = "Telegram bot credentials changed"
            success = "Configured Telegram provider; the bot token is private."
        else:
            if google_requirements is None:
                google_requirements = (
                    self._google_requirement_pairs()
                    or (("google-sheets", "read-write"),)
                )
            google_profile = (
                self.workspace.connector_provider_profiles().get(name, {})
            )
            client_json = self._collect_google_client_json(google_profile)
            self.workspace.save_connector_provider_secret(
                name, "oauth_client_json", client_json
            )
            scopes = self._google_scopes_for_requirements(
                google_requirements
            )
            if preserve_google_scopes:
                from zippergen.google_auth import normalize_google_scopes

                scopes = normalize_google_scopes((
                    *self._google_profile_scopes(
                        self.workspace.connector_provider_profiles().get(name)
                    ),
                    *scopes,
                ))
            service_names = ", ".join(
                sorted({
                    "Gmail" if kind == "gmail" else "Google Sheets"
                    for kind, _access in google_requirements
                })
            )
            try:
                authorized_user, authorization_ssh_host = (
                    self._authorize_google_provider(
                        client_json,
                        scopes=scopes,
                        service_names=service_names,
                        profile=google_profile,
                    )
                )
            except Exception as exc:
                raise SystemExit(str(exc)) from exc
            self.workspace.save_connector_provider_secret(
                name, "authorized_user_json", authorized_user
            )
            detail = "Google authorization changed"
            success = (
                "Configured Google provider; the OAuth client and token are "
                "stored privately."
            )
        self.workspace.save_connector_provider_profile(
            name,
            {
                "kind": "human-delivery" if name == "telegram" else "google",
                **(
                    {
                        "client_storage": "private Studio storage",
                        "scopes": json.dumps(list(scopes)),
                        **(
                            {
                                "authorization_ssh_host": (
                                    authorization_ssh_host
                                )
                            }
                            if authorization_ssh_host
                            else {}
                        ),
                    }
                    if name == "google"
                    else {}
                ),
                "check_status": "not checked",
                "check_detail": detail,
            },
        )
        self._success(success)
        if not self._check_connector_provider(name):
            raise SystemExit(
                f"Connector provider {name} was saved but is unavailable. "
                f"Fix it, then use 'connector provider check {name}'."
            )

    def _configure_connector_configuration(
        self,
        requested_name: str | None,
        *,
        edit: bool = False,
        provider_hint: str | None = None,
        kind_hint: str | None = None,
    ) -> str:
        configurations = self.workspace.connector_configurations()
        if edit:
            assert requested_name is not None
            name = self._connector_configuration_name(requested_name)
            existing = configurations[name]
        else:
            name = requested_name or ""
            existing = configurations.get(name, {})
        providers = self.workspace.connector_provider_profiles()
        if not providers:
            raise SystemExit(
                "No connector provider is configured. Use "
                "'connector provider configure telegram' or "
                "'connector provider configure google' first."
            )
        current_provider = str(
            existing.get("provider")
            or existing.get("kind")
            or next(iter(providers))
        )
        if provider_hint is not None:
            if provider_hint not in providers:
                raise SystemExit(
                    f"Connector provider is not configured: {provider_hint}."
                )
            provider = provider_hint
        elif len(providers) == 1:
            provider = next(iter(providers))
        else:
            provider = str(
                self._select(
                    "Connector providers",
                    list(providers),
                    prompt="Select provider",
                )
            )
        if not edit:
            name = requested_name or (
                "telegram-approvals"
                if provider == "telegram"
                else "google-sheet"
            )
            existing = configurations.get(name, {})
        if provider == "telegram":
            current_chat = str(existing.get("chat_id") or "")
            chat_id = self.input(
                f"Telegram chat id [{current_chat}]: "
                if current_chat
                else "Telegram chat id: "
            ).strip() or current_chat
            if not chat_id:
                raise SystemExit("Telegram chat id must not be empty.")
            record = {
                "provider": provider,
                "kind": "telegram",
                "chat_id": chat_id,
                "channel": f"telegram:{name}",
            }
        elif provider == "google":
            connector_kind = (
                kind_hint
                or str(existing.get("kind") or "")
                or str(
                    self._select(
                        "Google connector type",
                        ["google-sheets", "gmail"],
                        prompt="Select resource type",
                    )
                )
            )
            profile = providers.get("google", {})
            requirement_pairs = self._google_requirement_pairs(
                kinds=(connector_kind,)
            ) or ((connector_kind, "read-write"),)
            required_scopes = self._google_scopes_for_requirements(
                requirement_pairs
            )
            configured_scopes = self._google_profile_scopes(profile)
            if not self._google_scopes_cover(
                configured_scopes,
                required_scopes,
            ):
                self._info(
                    "Google authorization needs one additional service scope."
                )
                self._configure_connector_provider(
                    "google",
                    google_requirements=requirement_pairs,
                    preserve_google_scopes=True,
                )
            if connector_kind == "google-sheets":
                current_sheet = str(existing.get("spreadsheet_id") or "")
                sheet_value = self.input(
                    (
                        f"Google spreadsheet URL or ID [{current_sheet}]: "
                        if current_sheet
                        else "Google spreadsheet URL or ID: "
                    )
                ).strip() or current_sheet
                spreadsheet_id = self._google_spreadsheet_id(sheet_value)
                if not spreadsheet_id:
                    raise SystemExit(
                        "Google spreadsheet URL or ID must not be empty."
                    )
                current_tab = str(existing.get("tab") or "Sheet1")
                tab = (
                    self.input(f"Sheet tab [{current_tab}]: ").strip()
                    or current_tab
                )
                record = {
                    "provider": provider,
                    "kind": "google-sheets",
                    "spreadsheet_id": spreadsheet_id,
                    "tab": tab,
                    "resource": f"{spreadsheet_id}/{tab}",
                }
            elif connector_kind == "gmail":
                current_query = str(
                    existing.get("query") or "is:unread in:inbox"
                )
                query = (
                    self.input(f"Gmail search query [{current_query}]: ")
                    .strip()
                    or current_query
                )
                record = {
                    "provider": provider,
                    "kind": "gmail",
                    "account": "me",
                    "query": query,
                    "resource": f"me · {query}",
                }
            else:
                raise SystemExit(
                    "Google connector type must be google-sheets or gmail."
                )
        else:
            raise SystemExit(
                f"Connector provider {provider!r} is not supported."
            )
        self.workspace.save_connector_configuration(
            name,
            {
                **record,
                "check_status": "not checked",
                "check_detail": "configuration changed",
            },
        )
        self._success(f"Saved connector configuration {name}.")
        if not self._check_connector_configuration(name):
            raise SystemExit(
                f"Connector configuration {name} was saved but is "
                f"unavailable. Fix it, then use 'connector config check "
                f"{name}'."
            )
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
            current, workflow, module = self._current_context()
            _current, requirements = self._connector_requirements()
            bindings = self.workspace.connector_binding_profile(current)
            configurations = self.workspace.connector_configurations()
            completed: list[str] = []
            google_requirements = tuple(
                (requirement.kind, requirement.access)
                for requirement in requirements
                if requirement.kind in {"google-sheets", "gmail"}
            )
            if google_requirements:
                profile = self.workspace.connector_provider_profiles().get(
                    "google"
                )
                required_scopes = self._google_scopes_for_requirements(
                    google_requirements
                )
                configured_scopes = self._google_profile_scopes(profile)
                if profile is None or not self._google_scopes_cover(
                    configured_scopes,
                    required_scopes,
                ):
                    self._configure_connector_provider(
                        "google",
                        google_requirements=google_requirements,
                        preserve_google_scopes=True,
                    )
            for requirement in requirements:
                if requirement.name in bindings:
                    completed.append(
                        f"{requirement.name}={bindings[requirement.name]}"
                    )
                    continue
                provider = {
                    "google-sheets": "google",
                    "gmail": "google",
                }.get(requirement.kind)
                if provider is None:
                    raise SystemExit(
                        f"Guided setup is not available for "
                        f"{requirement.kind!r}. Use connector provider, config, "
                        "and bind explicitly."
                    )
                if (
                    provider
                    not in self.workspace.connector_provider_profiles()
                ):
                    self._configure_connector_provider(provider)
                matching = [
                    name
                    for name, value in configurations.items()
                    if value.get("kind") == requirement.kind
                ]
                if len(matching) == 1:
                    configuration_name = matching[0]
                elif len(matching) > 1:
                    configuration_name = str(
                        self._select(
                            f"Configurations for {requirement.name}",
                            matching,
                            prompt="Select configuration",
                        )
                    )
                else:
                    configuration_name = (
                        self._configure_connector_configuration(
                            requirement.name,
                            provider_hint=provider,
                            kind_hint=requirement.kind,
                        )
                    )
                    configurations = (
                        self.workspace.connector_configurations()
                    )
                self.workspace.bind_connector(
                    current,
                    requirement.name,
                    configuration_name,
                )
                completed.append(
                    f"{requirement.name}={configuration_name}"
                )

            human = self._human_action_lifelines(workflow, module)
            profile = self.workspace.connector_assignment_profile(current)
            lifelines = dict(profile["lifelines"])
            unassigned_human = [
                participant
                for participant in human
                if participant not in lifelines
            ]
            if unassigned_human:
                if (
                    "telegram"
                    not in self.workspace.connector_provider_profiles()
                ):
                    self._configure_connector_provider("telegram")
                configurations = self.workspace.connector_configurations()
                matching = [
                    name
                    for name, value in configurations.items()
                    if value.get("kind") == "telegram"
                ]
                if len(matching) == 1:
                    human_configuration = matching[0]
                elif len(matching) > 1:
                    human_configuration = str(
                        self._select(
                            "Human connector configurations",
                            matching,
                            prompt="Select configuration",
                        )
                    )
                else:
                    human_configuration = (
                        self._configure_connector_configuration(
                            "telegram-approvals",
                            provider_hint="telegram",
                        )
                    )
                for participant in unassigned_human:
                    lifelines[participant] = human_configuration
                self.workspace.save_connector_assignment_profile(
                    current,
                    lifelines=lifelines,
                    actions=profile["actions"],
                )
                completed.extend(
                    f"{participant}={human_configuration}"
                    for participant in unassigned_human
                )
            self._success(
                "Connector setup complete"
                + (
                    ": " + " · ".join(completed)
                    if completed
                    else ". This workflow has no connector requirements"
                )
                + "."
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
                "Use connector provider list, configure telegram|google, "
                "check [NAME|all], or remove NAME."
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
