# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false, reportArgumentType=false

"""Call/project intake workflow for local deployment.

The workflow watches an email inbox, accepts messages only from certified
senders, asks an LLM to classify/extract calls, replies with the extracted JSON,
and records new calls in a local CSV table. Corrections are handled by replying
to the JSON email with corrected fields, preferably keeping the same call_id.

Manual deployment:

    uv run python examples/call_intake_email_client.py --setup

    export ZIPPERGEN_CERTIFIED_SENDERS="alice@example.com,@trusted-lab.org"
    export ZIPPERGEN_CALL_INTAKE_RECIPIENTS="zippergen.sandbox+calls@gmail.com"
    export ZIPPERGEN_CALL_GMAIL_QUERY="is:unread in:inbox to:zippergen.sandbox+calls@gmail.com"
    export ZIPPERGEN_CALL_TABLE="$HOME/.zippergen/calls.csv"
    export ZIPPERGEN_CALL_SHEET_ID="<google-sheet-id>"
    export ZIPPERGEN_CALL_TABLE_TARGETS=both
    export ZIPPERGEN_CALL_INTAKE_SEND_MODE=send
    export ZIPPERGEN_CALL_INTAKE_MAX_EMAILS_PER_HOUR=10
    export ZIPPERGEN_CALL_INTAKE_POLL_SECONDS=60

    uv run zippergen run examples/call_intake.py:call_intake \
      --store "$HOME/.zippergen/runs/call-intake.sqlite" \
      --llm openai:gpt-4o \
      --services live \
      --llm-idle-timeout 300 \
      --timeout 0
"""

import csv
import hashlib
import html
import json
import os
import re
import time
from email.utils import getaddresses
from email.utils import parseaddr
from pathlib import Path

from zippergen import Lifeline, Var, effect, llm, pure, workflow


# ---------------------------------------------------------------------------
# Lifelines
# ---------------------------------------------------------------------------

Mailbox = Lifeline("Mailbox")
Gatekeeper = Lifeline("Gatekeeper")
Extractor = Lifeline("Extractor")
Table = Lifeline("Table")


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

email = Var("email", str, default="")
certified = Var("certified", bool, default=False)
expected_recipient = Var("expected_recipient", bool, default=False)
intake_kind = Var("intake_kind", str, default="")
call_json = Var("call_json", str, default="{}")
table_status = Var("table_status", str, default="")
response_status = Var("response_status", str, default="")
mail_status = Var("mail_status", str, default="")
intake_status = Var("intake_status", str, default="")
processed_count = Var("processed_count", int, default=0)
_ = Var("_", str, default="")


# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------

_email_client: object = None
_sheets_client: object = None
_fake_inbox: list[str] = []
_certified_senders: set[str] = set()
_intake_recipients: set[str] = set()
_message_limit = 2**31 - 1
_table_path: Path = Path.home() / ".zippergen" / "calls.csv"
_sheet_id = ""
_sheet_name = "Calls"
_table_targets: set[str] = {"csv"}
_response_log_path: Path = Path.home() / ".zippergen" / "call-intake-responses.jsonl"
_send_mode = "send"
_send_limit_per_hour = 10
_poll_seconds = 60.0


CALL_FIELDS = [
    "call_id",
    "status",
    "type",
    "title",
    "funding_organism",
    "domain_topic",
    "opening_date",
    "deadline",
    "amount_of_funding",
    "duration",
    "url",
    "summary",
    "extra_json",
    "source_sender",
    "source_subject",
    "source_message_id",
    "updated_at",
]

MATERIAL_CALL_FIELDS = [
    field for field in CALL_FIELDS
    if field not in {"call_id", "status", "source_sender", "source_subject", "source_message_id", "updated_at"}
]


DEFAULT_FAKE_INBOX = [
    """From: Alice Example <alice@example.com>
Subject: ERC Starting Grant call in formal methods
Message-ID: <fake-call-1@example.com>

Dear team,

The ERC Starting Grant call for formal methods and programming languages is open.
Deadline: 2026-10-15. Funding up to EUR 1.5M for five years.
More information: https://erc.europa.eu/apply-grant/starting-grant
"""
]


def _as_path(value: object, default: Path) -> Path:
    text = str(value or "").strip()
    return Path(text).expanduser() if text else default


def _parse_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    parsed = int(value)
    return parsed if parsed >= 0 else None


def _parse_send_limit(value: object) -> int:
    if value is None or value == "":
        return 10
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("send_limit_per_hour must be greater than zero.")
    if parsed > 10:
        raise ValueError("send_limit_per_hour may not exceed 10.")
    return parsed


def _split_senders(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = str(value).replace("\n", ",").split(",")
    return {str(item).strip().lower() for item in items if str(item).strip()}


def _parse_table_targets(value: object, sheet_id: str) -> set[str]:
    if value is None or value == "":
        return {"csv", "sheets"} if sheet_id else {"csv"}
    text = str(value).strip().lower()
    if text == "both":
        return {"csv", "sheets"}
    targets = {item.strip() for item in text.replace("\n", ",").split(",") if item.strip()}
    if not targets <= {"csv", "sheets"}:
        raise ValueError("table_targets must be 'csv', 'sheets', or 'both'.")
    if "sheets" in targets and not sheet_id:
        raise ValueError("table_targets includes 'sheets' but no sheet_id was configured.")
    return targets or {"csv"}


def _load_json_or_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return [line for line in text.splitlines() if line.strip()]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ValueError(f"Fake inbox file must contain a JSON list or non-empty lines: {path}")


def _reset_state() -> None:
    global _email_client, _sheets_client, _fake_inbox, _certified_senders, _intake_recipients
    global _message_limit, _table_path, _sheet_id, _sheet_name, _table_targets
    global _response_log_path, _send_mode, _send_limit_per_hour, _poll_seconds
    _email_client = None
    _sheets_client = None
    _fake_inbox = list(DEFAULT_FAKE_INBOX)
    _certified_senders = {"alice@example.com"}
    _intake_recipients = set()
    _message_limit = 2**31 - 1
    _table_path = Path.home() / ".zippergen" / "calls.csv"
    _sheet_id = ""
    _sheet_name = "Calls"
    _table_targets = {"csv"}
    _response_log_path = Path.home() / ".zippergen" / "call-intake-responses.jsonl"
    _send_mode = "send"
    _send_limit_per_hour = 10
    _poll_seconds = 60.0


def configure_call_intake(
    *,
    services: object = "fake",
    certified_senders: object = None,
    table_path: object = None,
    response_log_path: object = None,
    fake_inbox_path: object = None,
    send_mode: object = None,
    send_limit_per_hour: object = None,
    max_messages: object = None,
    poll_seconds: object = None,
    intake_recipients: object = None,
    sheet_id: object = None,
    sheet_name: object = None,
    table_targets: object = None,
) -> None:
    """Configure globals used by the deployment example and tests."""

    global _email_client, _sheets_client, _fake_inbox, _certified_senders, _intake_recipients
    global _message_limit, _table_path, _sheet_id, _sheet_name, _table_targets
    global _response_log_path, _send_mode, _send_limit_per_hour, _poll_seconds

    services_text = str(services or "fake")
    if services_text not in {"fake", "live"}:
        raise ValueError("services must be 'fake' or 'live'.")

    parsed_max_messages = _parse_int(max_messages)
    _message_limit = parsed_max_messages if parsed_max_messages is not None else 2**31 - 1
    _table_path = _as_path(table_path, Path(os.environ.get(
        "ZIPPERGEN_CALL_TABLE",
        str(Path.home() / ".zippergen" / "calls.csv"),
    )))
    raw_sheet_id = sheet_id
    if raw_sheet_id is None:
        raw_sheet_id = os.environ.get("ZIPPERGEN_CALL_SHEET_ID", "")
    _sheet_id = str(raw_sheet_id or "").strip()
    _sheet_name = str(
        sheet_name
        or os.environ.get("ZIPPERGEN_CALL_SHEET_NAME", "Calls")
        or "Calls"
    ).strip()
    _table_targets = _parse_table_targets(
        table_targets if table_targets is not None else os.environ.get("ZIPPERGEN_CALL_TABLE_TARGETS"),
        _sheet_id,
    )
    _response_log_path = _as_path(response_log_path, Path(os.environ.get(
        "ZIPPERGEN_CALL_INTAKE_RESPONSE_LOG",
        str(Path.home() / ".zippergen" / "call-intake-responses.jsonl"),
    )))
    _send_mode = str(send_mode or os.environ.get("ZIPPERGEN_CALL_INTAKE_SEND_MODE", "send")).strip().lower()
    if _send_mode not in {"draft", "send", "log"}:
        raise ValueError("send_mode must be 'draft', 'send', or 'log'.")
    raw_limit = send_limit_per_hour
    if raw_limit is None:
        raw_limit = os.environ.get("ZIPPERGEN_CALL_INTAKE_MAX_EMAILS_PER_HOUR", "10")
    _send_limit_per_hour = _parse_send_limit(raw_limit)
    _poll_seconds = float(poll_seconds or os.environ.get("ZIPPERGEN_CALL_INTAKE_POLL_SECONDS", "60"))

    raw_senders = certified_senders
    if raw_senders is None:
        raw_senders = os.environ.get("ZIPPERGEN_CERTIFIED_SENDERS")
    _certified_senders = _split_senders(raw_senders)
    if not _certified_senders and services_text == "fake":
        _certified_senders = {"alice@example.com"}

    raw_recipients = intake_recipients
    if raw_recipients is None:
        raw_recipients = os.environ.get(
            "ZIPPERGEN_CALL_INTAKE_RECIPIENTS",
            os.environ.get("ZIPPERGEN_CALL_INTAKE_ADDRESS"),
        )
    _intake_recipients = _split_senders(raw_recipients)

    if services_text == "live":
        _email_client = _load_service_module("call_intake_email_client")
        _sheets_client = _load_service_module("call_intake_sheets_client") if "sheets" in _table_targets else None
        _fake_inbox = []
    else:
        _email_client = None
        _sheets_client = None
        inbox_path = _as_path(fake_inbox_path, Path(os.environ.get("ZIPPERGEN_CALL_INTAKE_FAKE_INBOX", "")))
        _fake_inbox = _load_json_or_lines(inbox_path) if str(inbox_path) != "." else []
        if not _fake_inbox:
            _fake_inbox = list(DEFAULT_FAKE_INBOX)


def reset_for_tests(
    *,
    fake_inbox: list[str] | None = None,
    certified_senders: object = "alice@example.com",
    table_path: object = None,
    response_log_path: object = None,
    send_mode: object = "log",
    send_limit_per_hour: object = 10,
    max_messages: object = 1,
    intake_recipients: object = None,
    sheet_id: object = None,
    sheet_name: object = None,
    table_targets: object = None,
) -> None:
    configure_call_intake(
        services="fake",
        certified_senders=certified_senders,
        table_path=table_path,
        response_log_path=response_log_path,
        send_mode=send_mode,
        send_limit_per_hour=send_limit_per_hour,
        max_messages=max_messages,
        intake_recipients=intake_recipients,
        sheet_id=sheet_id,
        sheet_name=sheet_name,
        table_targets=table_targets,
    )
    global _fake_inbox
    _fake_inbox = list(fake_inbox or [])


def _load_service_module(name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parent / f"{name}.py"
    )
    assert spec and spec.loader, f"Could not load {name}.py"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def zippergen_setup(config) -> None:
    """Hook called by ``zippergen run`` before configuring the workflow."""

    intake_recipients = config.option("recipient", None)
    if intake_recipients is None:
        intake_recipients = config.option("recipients", None)
    sheet_id = config.option("sheet_id", None)
    if sheet_id is None:
        sheet_id = config.option("sheet", None)
    configure_call_intake(
        services=config.option("services", "fake"),
        certified_senders=config.option("certified", None),
        table_path=config.option("table", None),
        response_log_path=config.option("response_log", None),
        send_mode=config.option("send_mode", None),
        send_limit_per_hour=config.option("send_limit_per_hour", None),
        max_messages=config.option("max_messages", None),
        poll_seconds=config.option("poll_seconds", None),
        intake_recipients=intake_recipients,
        sheet_id=sheet_id,
        sheet_name=config.option("sheet_name", None),
        table_targets=config.option("table_targets", None),
    )


# ---------------------------------------------------------------------------
# Email and JSON helpers
# ---------------------------------------------------------------------------

def _parse_email_text(text: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    raw_headers, sep, body = text.partition("\n\n")
    if not sep:
        body = text
    current = ""
    for line in raw_headers.splitlines():
        if line.startswith((" ", "\t")) and current:
            headers[current] = (headers[current] + " " + line.strip()).strip()
            continue
        name, colon, value = line.partition(":")
        if colon:
            current = name.strip().lower()
            headers[current] = value.strip()
    sender = headers.get("from", "")
    return {
        "sender": sender,
        "sender_email": parseaddr(sender)[1].lower(),
        "to": headers.get("to", ""),
        "cc": headers.get("cc", ""),
        "delivered_to": headers.get("delivered-to", ""),
        "x_original_to": headers.get("x-original-to", ""),
        "envelope_to": headers.get("envelope-to", ""),
        "subject": headers.get("subject", ""),
        "message_id": headers.get("message-id", ""),
        "thread_id": headers.get("thread-id", ""),
        "gmail_id": headers.get("gmail-id", ""),
        "in_reply_to": headers.get("in-reply-to", ""),
        "references": headers.get("references", ""),
        "body": body.strip(),
    }


def _format_email(meta: dict) -> str:
    lines = [
        f"From: {meta.get('sender', '')}",
        f"To: {meta.get('to', '')}",
        f"Subject: {meta.get('subject', '(no subject)')}",
    ]
    for source_key, header in [
        ("cc", "Cc"),
        ("delivered_to", "Delivered-To"),
        ("x_original_to", "X-Original-To"),
        ("envelope_to", "Envelope-To"),
        ("id", "Gmail-ID"),
        ("thread_id", "Thread-ID"),
        ("message_id", "Message-ID"),
        ("in_reply_to", "In-Reply-To"),
        ("references", "References"),
    ]:
        value = meta.get(source_key)
        if value:
            lines.append(f"{header}: {value}")
    return "\n".join(lines) + "\n\n" + str(meta.get("body", "")).strip()


def _looks_like_email_address(address: str) -> bool:
    local, sep, domain = address.partition("@")
    return bool(local and sep and domain)


def _header_addresses(*values: str) -> set[str]:
    return {
        address.strip().lower()
        for _name, address in getaddresses([value for value in values if value])
        if _looks_like_email_address(address.strip().lower())
    }


def _recipient_addresses(meta: dict[str, str]) -> set[str]:
    return _header_addresses(
        meta.get("to", ""),
        meta.get("cc", ""),
        meta.get("delivered_to", ""),
        meta.get("x_original_to", ""),
        meta.get("envelope_to", ""),
    )


def _validated_reply_recipient(meta: dict[str, str]) -> str:
    sender = str(meta.get("sender", "")).strip()
    parsed_sender = parseaddr(sender)[1].strip().lower()
    sender_email = str(meta.get("sender_email", "")).strip().lower()
    if not sender_email:
        sender_email = parsed_sender
    if not _looks_like_email_address(sender_email):
        raise ValueError("Refusing to send response: sender email address is not valid.")
    if parsed_sender and parsed_sender != sender_email:
        raise ValueError(
            "Refusing to send response: parsed sender address does not match "
            "the response recipient."
        )
    return sender_email


def _intake_reply_address() -> str:
    explicit = os.environ.get("ZIPPERGEN_CALL_INTAKE_REPLY_TO", "").strip().lower()
    if explicit and _looks_like_email_address(explicit):
        return explicit
    return sorted(_intake_recipients)[0] if _intake_recipients else ""


def _extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    decoder = json.JSONDecoder()
    for idx, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(cleaned[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("No JSON object found.")


CORRECTION_FIELD_CANONICAL = {
    "call_id": "call_id",
    "status": "status",
    "type": "type",
    "call_type": "type",
    "title": "title",
    "name": "title",
    "funding_organism": "funding_organism",
    "funding_agency": "funding_organism",
    "funder": "funding_organism",
    "domain_topic": "domain_topic",
    "domain": "domain_topic",
    "topic": "domain_topic",
    "opening_date": "opening_date",
    "opens": "opening_date",
    "deadline": "deadline",
    "submission_deadline": "deadline",
    "application_deadline": "deadline",
    "due_date": "deadline",
    "closing_date": "deadline",
    "closing_deadline": "deadline",
    "submission_due_date": "deadline",
    "date_limite": "deadline",
    "date_limite_soumission": "deadline",
    "date_limite_candidature": "deadline",
    "amount_of_funding": "amount_of_funding",
    "amount": "amount_of_funding",
    "funding_amount": "amount_of_funding",
    "duration": "duration",
    "project_duration": "duration",
    "url": "url",
    "link": "url",
    "summary": "summary",
    "description": "summary",
    "notes": "summary",
}

CORRECTION_FIELD_ALIASES = set(CORRECTION_FIELD_CANONICAL)


def _parse_fragment_value(raw: str) -> object:
    text = raw.strip().rstrip(",").strip()
    try:
        value, _end = json.JSONDecoder().raw_decode(text)
        return value
    except json.JSONDecodeError:
        pass
    while text.endswith("}"):
        text = text[:-1].rstrip().rstrip(",").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        else:
            text = text.strip("\"'")
        return re.sub(r"\s*\n\s*", " ", text).strip()


def _extract_json_like_fields(text: str) -> dict:
    key_pattern = re.compile(
        r'(?m)(?P<prefix>^|[,{]\s*)"?(?P<key>[A-Za-z_][A-Za-z0-9_]*)"?\s*:'
    )
    matches = [
        match for match in key_pattern.finditer(text)
        if match.group("key") in CORRECTION_FIELD_ALIASES
    ]
    fields: dict[str, object] = {}
    seen_canonical: set[str] = set()
    for index, match in enumerate(matches):
        key = match.group("key")
        canonical = CORRECTION_FIELD_CANONICAL[key]
        if canonical in seen_canonical:
            continue
        value_start = match.end()
        value_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        raw_value = text[value_start:value_end].strip()
        if raw_value:
            fields[key] = _parse_fragment_value(raw_value)
            seen_canonical.add(canonical)
    if fields:
        return fields
    raise ValueError("No JSON-like correction fields found.")


def _html_to_text(text: str) -> str:
    if re.search(r"(?is)<(?:html|body|div|br|p|blockquote|span|table|tr|td|a)\b", text):
        text = re.sub(r"(?is)<(br|/p|/div|/li|/tr)\b[^>]*>", "\n", text)
        text = re.sub(r"(?is)<(script|style)\b[^>]*>.*?</\1>", "", text)
        text = re.sub(r"(?is)<[^>]+>", "", text)
    return html.unescape(text).replace("\xa0", " ")


def _looks_like_reply_quote_header(line: str) -> bool:
    text = re.sub(r"\s+", " ", line.strip().lower())
    if not text:
        return False
    if text.startswith(("-----original message-----", "---------- forwarded message")):
        return True
    reply_openers = ("on ", "le ", "am ", "el ", "il ", "op ")
    reply_markers = ("wrote:", "a écrit", "schrieb", "escribió", "ha scritto")
    return text.startswith(reply_openers) and any(marker in text for marker in reply_markers)


def _leading_reply_text(text: str) -> tuple[str, str]:
    plain = _html_to_text(text).strip()
    lines: list[str] = []
    for line in plain.splitlines():
        if _looks_like_reply_quote_header(line):
            break
        if line.lstrip().startswith(">"):
            continue
        lines.append(line)
    leading = "\n".join(lines).strip()
    return leading or plain, plain


def _extract_correction_data_from_text(text: str) -> dict:
    stripped = text.strip()
    if not stripped:
        raise ValueError("No correction text found.")
    parsers = (
        (_extract_json_object, _extract_json_like_fields)
        if stripped.startswith("{")
        else (_extract_json_like_fields, _extract_json_object)
    )
    for parser in parsers:
        try:
            return parser(stripped)
        except ValueError:
            continue
    raise ValueError("No sender correction object found.")


def _extract_sender_correction_object(text: str) -> dict:
    leading, plain = _leading_reply_text(text)
    try:
        return _extract_correction_data_from_text(leading)
    except ValueError:
        if leading != plain:
            raise
    return _extract_correction_data_from_text(plain)


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _sanitize_call_id(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-._")
    return text[:80]


def _find_call_id(email_text: str, data: dict) -> str:
    explicit = _sanitize_call_id(data.get("call_id"))
    if explicit:
        return explicit
    match = re.search(r"\bcall_[A-Za-z0-9_.-]{1,80}\b", email_text)
    if match:
        return _sanitize_call_id(match.group(0))
    meta = _parse_email_text(email_text)
    key = "\n".join([
        meta.get("sender_email", ""),
        meta.get("subject", ""),
        str(data.get("title") or ""),
        str(data.get("url") or ""),
        _string_field(
            data,
            "deadline",
            "submission_deadline",
            "application_deadline",
            "due_date",
            "closing_date",
            "date_limite",
        ),
    ])
    return "call_" + _short_hash(key)


def _string_field(data: dict, *names: str) -> str:
    for name in names:
        value = data.get(name)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _normalise_payload(email_text: str, raw_json: str, *, status: str) -> str:
    try:
        data = _extract_json_object(raw_json)
    except ValueError:
        stripped = raw_json.strip()
        try:
            data = _extract_json_like_fields(stripped)
        except ValueError:
            if stripped.startswith("{"):
                data = {"extra": {"unparsed": stripped}}
            else:
                data = {"summary": stripped, "extra": {"unparsed": stripped}}

    meta = _parse_email_text(email_text)
    standard = {
        "call_id": _find_call_id(email_text, data),
        "status": status,
        "type": _string_field(data, "type", "call_type"),
        "title": _string_field(data, "title", "name"),
        "funding_organism": _string_field(data, "funding_organism", "funding_agency", "funder"),
        "domain_topic": _string_field(data, "domain_topic", "domain", "topic"),
        "opening_date": _string_field(data, "opening_date", "opens"),
        "deadline": _string_field(
            data,
            "deadline",
            "submission_deadline",
            "application_deadline",
            "due_date",
            "closing_date",
            "closing_deadline",
            "submission_due_date",
            "date_limite",
            "date_limite_soumission",
            "date_limite_candidature",
        ),
        "amount_of_funding": _string_field(data, "amount_of_funding", "amount", "funding_amount"),
        "duration": _string_field(data, "duration", "project_duration"),
        "url": _string_field(data, "url", "link"),
        "summary": _string_field(data, "summary", "description", "notes"),
        "source_sender": meta.get("sender_email", ""),
        "source_subject": meta.get("subject", ""),
        "source_message_id": (
            meta.get("message_id", "")
            or meta.get("thread_id", "")
            or meta.get("gmail_id", "")
        ),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    known = {
        "call_id", "status", "type", "call_type", "title", "name",
        "funding_organism", "funding_agency", "funder", "domain_topic",
        "domain", "topic", "opening_date", "opens", "deadline",
        "submission_deadline", "application_deadline", "due_date",
        "closing_date", "closing_deadline", "submission_due_date",
        "date_limite", "date_limite_soumission", "date_limite_candidature",
        "amount_of_funding", "amount", "funding_amount", "duration",
        "project_duration", "url", "link", "summary", "description", "notes",
    }
    extra = {key: value for key, value in data.items() if key not in known}
    standard["extra_json"] = json.dumps(extra, sort_keys=True) if extra else ""
    return json.dumps(standard, sort_keys=True)


def _record_from_json(call_json_text: str) -> dict[str, str]:
    data = _extract_json_object(call_json_text)
    return {field: str(data.get(field, "") or "") for field in CALL_FIELDS}


def _read_table(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [
            {field: str(row.get(field, "") or "") for field in CALL_FIELDS}
            for row in csv.DictReader(f)
        ]


def _write_table(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CALL_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CALL_FIELDS})


def _sheet_enabled() -> bool:
    return "sheets" in _table_targets and bool(_sheet_id)


def _read_records() -> list[dict[str, str]]:
    if _sheet_enabled():
        if _sheets_client is None:
            raise RuntimeError("Google Sheets table target is configured, but the Sheets client is not loaded.")
        return _sheets_client.read_rows(_sheet_id, _sheet_name, CALL_FIELDS)  # type: ignore[union-attr]
    return _read_table(_table_path)


def _write_records(rows: list[dict[str, str]]) -> None:
    if "csv" in _table_targets:
        _write_table(_table_path, rows)
    if _sheet_enabled():
        if _sheets_client is None:
            raise RuntimeError("Google Sheets table target is configured, but the Sheets client is not loaded.")
        _sheets_client.write_rows(_sheet_id, _sheet_name, CALL_FIELDS, rows)  # type: ignore[union-attr]


def _table_location_text() -> str:
    parts: list[str] = []
    if "csv" in _table_targets:
        parts.append(str(_table_path))
    if _sheet_enabled():
        parts.append(f"Google Sheet {_sheet_id}/{_sheet_name}")
    return " and ".join(parts) if parts else str(_table_path)


def _merge_row(existing: dict[str, str], incoming: dict[str, str]) -> dict[str, str]:
    merged = dict(existing)
    for field in CALL_FIELDS:
        value = incoming.get(field, "")
        if value or field in {"status", "updated_at"}:
            merged[field] = value
    return merged


def _material_correction_fields(existing: dict[str, str], incoming: dict[str, str]) -> list[str]:
    return [
        field for field in MATERIAL_CALL_FIELDS
        if incoming.get(field, "") and incoming.get(field, "") != existing.get(field, "")
    ]


def _status_call_id(table_status_text: str) -> str:
    parts = table_status_text.split(":", 2)
    return parts[1] if len(parts) >= 2 else ""


def _status_changed_fields(table_status_text: str) -> list[str]:
    if not table_status_text.startswith("updated:"):
        return []
    parts = table_status_text.split(":", 2)
    if len(parts) < 3:
        return []
    return [field for field in parts[2].split(",") if field]


def _record_for_call_id(call_id: str) -> dict[str, str] | None:
    if not call_id:
        return None
    for row in _read_records():
        if row.get("call_id") == call_id:
            return row
    return None


def _same_source_message(existing: dict[str, str], incoming: dict[str, str]) -> bool:
    source_id = incoming.get("source_message_id", "")
    return bool(source_id) and existing.get("source_message_id", "") == source_id


def _append_response_log_once(key: str, payload: dict) -> bool:
    _response_log_path.parent.mkdir(parents=True, exist_ok=True)
    if _response_log_contains(key):
        return False
    with _response_log_path.open("a") as f:
        f.write(json.dumps({"key": key, **payload}, sort_keys=True) + "\n")
    return True


def _response_log_records() -> list[dict]:
    if not _response_log_path.exists():
        return []
    records: list[dict] = []
    for line in _response_log_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _response_log_contains(key: str) -> bool:
    return any(str(record.get("key", "")) == key for record in _response_log_records())


def _response_log_contains_source(message_key: str) -> bool:
    if not message_key:
        return False
    return any(str(record.get("source_message_key", "")) == message_key for record in _response_log_records())


def _recent_send_timestamps(now: float) -> list[float]:
    cutoff = now - 3600.0
    timestamps: list[float] = []
    for record in _response_log_records():
        if record.get("mode") != "send":
            continue
        raw = record.get("sent_at")
        if not isinstance(raw, (int, float)):
            continue
        stamp = float(raw)
        if stamp > cutoff:
            timestamps.append(stamp)
    return sorted(timestamps)


def _send_rate_limit_delay(now: float | None = None) -> float:
    now = time.time() if now is None else now
    timestamps = _recent_send_timestamps(now)
    if len(timestamps) < _send_limit_per_hour:
        return 0.0
    oldest_blocking_send = timestamps[-_send_limit_per_hour]
    return max(0.0, oldest_blocking_send + 3600.0 - now)


def _wait_for_send_rate_limit() -> None:
    while True:
        delay = _send_rate_limit_delay()
        if delay <= 0:
            return
        wait_for = min(delay, 60.0)
        print(
            f"[CallIntake] Email send limit reached "
            f"({_send_limit_per_hour}/hour); waiting {wait_for:.0f}s."
        )
        time.sleep(wait_for)


# ---------------------------------------------------------------------------
# Polling and side effects
# ---------------------------------------------------------------------------

def _mailbox_retry_delay() -> float:
    return min(max(_poll_seconds, 0.0), 60.0)


def _log_mailbox_api_error(operation: str, exc: Exception) -> None:
    print(f"[CallIntake] Gmail {operation} failed ({type(exc).__name__}: {exc}); retrying.")


def mail_present() -> bool:
    if _email_client is not None:
        try:
            return _email_client.count_unread() > 0  # type: ignore[union-attr]
        except Exception as exc:
            _log_mailbox_api_error("poll", exc)
            return False
    return bool(_fake_inbox)


@effect
def pop_pending_email() -> str:
    if _email_client is not None:
        while True:
            try:
                meta = _email_client.fetch_one_unread()  # type: ignore[union-attr]
            except Exception as exc:
                _log_mailbox_api_error("fetch", exc)
                time.sleep(_mailbox_retry_delay())
                continue
            if meta is None:
                return ""
            return _format_email(meta)
    return _fake_inbox.pop(0) if _fake_inbox else ""


@pure
def increment_processed_count(count: int) -> int:
    return count + 1


@effect(visible=False)
def wait_briefly() -> str:
    time.sleep(_poll_seconds)
    return ""


@pure
def is_certified_sender(email: str) -> bool:
    sender = _parse_email_text(email).get("sender_email", "")
    if "*" in _certified_senders:
        return True
    for allowed in _certified_senders:
        if allowed.startswith("@") and sender.endswith(allowed):
            return True
        if sender == allowed:
            return True
    return False


@pure
def is_expected_intake_recipient(email: str) -> bool:
    if not _intake_recipients:
        return True
    return bool(_recipient_addresses(_parse_email_text(email)) & _intake_recipients)


@pure
def normalize_intake_kind(intake_kind: str) -> str:
    text = intake_kind.strip().lower()
    if "correction" in text or text in {"update", "corrected"}:
        return "correction"
    if "call" in text or text in {"project", "position", "funding"}:
        return "call"
    return "other"


@pure
def normalize_call_json(email: str, call_json: str) -> str:
    return _normalise_payload(email, call_json, status="new")


@pure
def normalize_correction_json(email: str, call_json: str) -> str:
    body = _parse_email_text(email).get("body", "")
    try:
        explicit_json = _extract_sender_correction_object(body)
    except ValueError:
        source_json = call_json
    else:
        source_json = json.dumps(explicit_json, sort_keys=True)
    return _normalise_payload(email, source_json, status="corrected")


@effect
def insert_call_record(email: str, call_json: str) -> str:
    incoming = _record_from_json(call_json)
    rows = _read_records()
    incoming_source_id = incoming.get("source_message_id", "")
    if incoming_source_id:
        for row in rows:
            if row.get("source_message_id", "") == incoming_source_id:
                existing_id = row.get("call_id") or incoming["call_id"]
                print(f"[CallIntake] {existing_id} is already recorded from this message")
                return f"created:{existing_id}"
    for row in rows:
        if row.get("call_id") == incoming["call_id"]:
            if _same_source_message(row, incoming):
                print(f"[CallIntake] {incoming['call_id']} is already recorded from this message")
                return f"created:{incoming['call_id']}"
            print(f"[CallIntake] Duplicate {incoming['call_id']} ignored in {_table_location_text()}")
            return f"duplicate:{incoming['call_id']}"
    rows.append(incoming)
    _write_records(rows)
    print(f"[CallIntake] created {incoming['call_id']} in {_table_location_text()}")
    return f"created:{incoming['call_id']}"


@effect
def apply_call_correction(email: str, call_json: str) -> str:
    incoming = _record_from_json(call_json)
    rows = _read_records()
    for index, row in enumerate(rows):
        if row.get("call_id") == incoming["call_id"]:
            changed_fields = _material_correction_fields(row, incoming)
            if not changed_fields:
                print(f"[CallIntake] Correction for {incoming['call_id']} contained no table changes")
                return f"unchanged:{incoming['call_id']}"
            rows[index] = _merge_row(row, incoming)
            _write_records(rows)
            fields_text = ", ".join(changed_fields)
            print(f"[CallIntake] updated {incoming['call_id']} ({fields_text}) in {_table_location_text()}")
            return f"updated:{incoming['call_id']}:{','.join(changed_fields)}"
    print(f"[CallIntake] Correction for missing {incoming['call_id']} ignored")
    return f"missing:{incoming['call_id']}"


@effect
def record_non_call(email: str) -> str:
    meta = _parse_email_text(email)
    print(f"[CallIntake] Ignored non-call from {meta.get('sender_email', '')}: {meta.get('subject', '')}")
    return "ignored_non_call"


@effect
def ignore_uncertified(email: str) -> str:
    meta = _parse_email_text(email)
    print(f"[CallIntake] Ignored uncertified sender {meta.get('sender_email', '')}")
    if _email_client is not None:
        _email_client.mark_processed(meta)  # type: ignore[union-attr]
    return "ignored_uncertified"


@effect
def ignore_unexpected_recipient(email: str) -> str:
    meta = _parse_email_text(email)
    found = ", ".join(sorted(_recipient_addresses(meta))) or "(none)"
    expected = ", ".join(sorted(_intake_recipients)) or "(none)"
    print(f"[CallIntake] Ignored message for {found}; expected {expected}")
    if _email_client is not None:
        _email_client.mark_processed(meta)  # type: ignore[union-attr]
    return "ignored_unexpected_recipient"


def _response_kind(table_status_text: str, *, correction: bool) -> str:
    if table_status_text.startswith("duplicate:"):
        return "duplicate"
    if correction and table_status_text.startswith("missing:"):
        return "missing_correction"
    if correction and table_status_text.startswith("unchanged:"):
        return "unchanged_correction"
    if correction:
        return "correction"
    return "extraction"


def _response_subject_prefix(kind: str) -> str:
    if kind == "duplicate":
        return "Call already recorded"
    if kind == "missing_correction":
        return "Call not found"
    if kind == "unchanged_correction":
        return "No call changes"
    if kind == "correction":
        return "Updated call JSON"
    return "Extracted call JSON"


def _response_heading(kind: str) -> str:
    if kind == "duplicate":
        return "This call already exists in the table, so I did not add or modify a row."
    if kind == "missing_correction":
        return (
            "I could not find an existing call with this call_id, so I did not "
            "modify the table. Please reply with the correct call_id or send it "
            "as a new call."
        )
    if kind == "unchanged_correction":
        return (
            "I found this call_id, but I did not detect any changed table fields, "
            "so I did not modify the table."
        )
    if kind == "correction":
        return "I updated the call table with this corrected JSON."
    return "I recorded this call with the following JSON."


def _response_body(
    call_json_text: str,
    table_status_text: str,
    *,
    correction: bool,
    data: dict | None = None,
) -> str:
    if data is None:
        data = _response_data(call_json_text, table_status_text, correction=correction)
    pretty = json.dumps(data, indent=2, sort_keys=True)
    kind = _response_kind(table_status_text, correction=correction)
    heading = _response_heading(kind)
    changed_fields = _status_changed_fields(table_status_text)
    changed_line = f"\nChanged fields: {', '.join(changed_fields)}\n" if changed_fields else ""
    reply_to = _intake_reply_address()
    correction_instruction = (
        f"If anything is wrong, reply to {reply_to} with corrected JSON. "
        if reply_to
        else "If anything is wrong, reply to this email with corrected JSON. "
    )
    return (
        f"Hello,\n\n{heading}\n\n"
        f"{pretty}\n\n"
        f"{changed_line}"
        f"{correction_instruction}Please keep the call_id unchanged.\n\n"
        f"Table status: {table_status_text}\n"
    )


def _response_data(call_json_text: str, table_status_text: str, *, correction: bool) -> dict:
    data = _extract_json_object(call_json_text)
    if not correction or table_status_text.startswith("missing:"):
        return data
    call_id = _status_call_id(table_status_text) or str(data.get("call_id", ""))
    row = _record_for_call_id(call_id)
    return row if row is not None else data


def _send_response(email: str, call_json_text: str, table_status_text: str, *, correction: bool) -> str:
    meta = _parse_email_text(email)
    recipient = _validated_reply_recipient(meta)
    data = _response_data(call_json_text, table_status_text, correction=correction)
    call_id = str(data.get("call_id", "call"))
    purpose = _response_kind(table_status_text, correction=correction)
    message_key = meta.get("message_id", "") or meta.get("thread_id", "") or meta.get("gmail_id", "")
    response_json_text = json.dumps(data, sort_keys=True)
    key = _short_hash("\n".join([purpose, message_key, call_id, table_status_text, response_json_text]))
    subject_prefix = _response_subject_prefix(purpose)
    subject = f"{subject_prefix}: {call_id}"
    body = _response_body(call_json_text, table_status_text, correction=correction, data=data)

    if _response_log_contains(key) or _response_log_contains_source(message_key):
        return f"already_recorded:{call_id}"

    payload = {
        "call_id": call_id,
        "mode": _send_mode,
        "purpose": purpose,
        "recipient": recipient,
        "subject": subject,
        "body": body,
        "changed_fields": _status_changed_fields(table_status_text),
        "source_message_key": message_key,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    if _email_client is not None and _send_mode in {"draft", "send"}:
        reply_meta = {**meta, "sender_email": recipient}
        if _send_mode == "send":
            _wait_for_send_rate_limit()
            external_id = _email_client.send_email(reply_meta, subject, body)  # type: ignore[union-attr]
            _append_response_log_once(
                key,
                {**payload, "mode": "send", "external_id": external_id, "sent_at": time.time()},
            )
            return f"send:{external_id}"
        else:
            external_id = _email_client.create_draft(reply_meta, subject, body)  # type: ignore[union-attr]
            _append_response_log_once(key, {**payload, "mode": "draft", "external_id": external_id})
            return f"draft:{external_id}"

    _append_response_log_once(key, {**payload, "mode": "log"})
    print(f"[CallIntake] Response for {call_id} logged at {_response_log_path}")
    return f"logged:{call_id}"


@effect
def send_call_json_response(email: str, call_json: str, table_status: str) -> str:
    return _send_response(email, call_json, table_status, correction=False)


@effect
def send_correction_response(email: str, call_json: str, table_status: str) -> str:
    return _send_response(email, call_json, table_status, correction=True)


@effect
def finish_email(email: str, status: str) -> str:
    meta = _parse_email_text(email)
    if _email_client is not None:
        _email_client.mark_processed(meta)  # type: ignore[union-attr]
    return f"processed:{status}"


@pure
def intake_finished(count: int) -> str:
    return f"processed:{count}"


# ---------------------------------------------------------------------------
# LLM actions
# ---------------------------------------------------------------------------

@llm(
    system="""
You classify emails for a research call intake system.
Return exactly one label:
  call       - a call for projects, funding, grants, positions, fellowships, jobs, proposals, or applications
  correction - a reply correcting previously extracted call JSON
  other      - anything else
Return only the label.
""".strip(),
    user="{email}",
    parse="text",
    outputs=(("intake_kind", str),),
)
def classify_intake(email: str) -> None: ...


@llm(
    system="""
Extract call details from the email.
Return only one JSON object. Use empty strings for unknown fields.

Schema:
{
  "type": "project | position | fellowship | grant | other",
  "title": "",
  "funding_organism": "",
  "domain_topic": "",
  "opening_date": "",
  "deadline": "",
  "amount_of_funding": "",
  "duration": "",
  "url": "",
  "summary": "",
  "extra": {}
}

Map submission deadline, application deadline, closing date, and date limite de candidature/soumission to "deadline".
""".strip(),
    user="{email}",
    parse="text",
    outputs=(("call_json", str),),
)
def extract_call_json(email: str) -> None: ...


@llm(
    system="""
Extract corrected call details from this reply.
Return only one JSON object.

Rules:
- If the sender wrote a JSON object, copy its fields and values exactly.
- Do not reinterpret, normalize, or change date/time values from sender-written JSON.
- Use only the new text written by the sender.
- Ignore quoted previous emails, quoted ZipperGen responses, and lines starting with ">".
- Preserve any call_id present in the email.
- Include only corrected or clearly restated fields.
- Use the same field names as the original call JSON when possible.
- Map submission deadline, application deadline, closing date, and date limite de candidature/soumission to "deadline".
""".strip(),
    user="{email}",
    parse="text",
    outputs=(("call_json", str),),
)
def extract_correction_json(email: str) -> None: ...


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

@workflow
def call_intake() -> str:
    while (processed_count < _message_limit) @ Mailbox:
        if mail_present() @ Mailbox:
            Mailbox: email = pop_pending_email()
            Mailbox(email) >> Gatekeeper(email)
            Gatekeeper: expected_recipient = is_expected_intake_recipient(email)

            if expected_recipient @ Gatekeeper:
                Gatekeeper: certified = is_certified_sender(email)

                if certified @ Gatekeeper:
                    Gatekeeper(email) >> Extractor(email)
                    Extractor: intake_kind = classify_intake(email)
                    Extractor: intake_kind = normalize_intake_kind(intake_kind)

                    if (intake_kind == "call") @ Extractor:
                        Extractor: call_json = extract_call_json(email)
                        Extractor: call_json = normalize_call_json(email, call_json)
                        Extractor(email, call_json) >> Table(email, call_json)
                        Table: table_status = insert_call_record(email, call_json)
                        Table(email, call_json, table_status) >> Mailbox(email, call_json, table_status)
                        Mailbox: response_status = send_call_json_response(email, call_json, table_status)
                        Mailbox: mail_status = finish_email(email, response_status)
                        Mailbox: processed_count = increment_processed_count(processed_count)

                    elif (intake_kind == "correction") @ Extractor:
                        Extractor: call_json = extract_correction_json(email)
                        Extractor: call_json = normalize_correction_json(email, call_json)
                        Extractor(email, call_json) >> Table(email, call_json)
                        Table: table_status = apply_call_correction(email, call_json)
                        Table(email, call_json, table_status) >> Mailbox(email, call_json, table_status)
                        Mailbox: response_status = send_correction_response(email, call_json, table_status)
                        Mailbox: mail_status = finish_email(email, response_status)
                        Mailbox: processed_count = increment_processed_count(processed_count)

                    else:
                        Extractor(email) >> Table(email)
                        Table: table_status = record_non_call(email)
                        Table(email, table_status) >> Mailbox(email, table_status)
                        Mailbox: mail_status = finish_email(email, table_status)
                        Mailbox: processed_count = increment_processed_count(processed_count)
                else:
                    Gatekeeper(email) >> Mailbox(email)
                    Mailbox: mail_status = ignore_uncertified(email)
                    Mailbox: processed_count = increment_processed_count(processed_count)
            else:
                Gatekeeper(email) >> Mailbox(email)
                Mailbox: mail_status = ignore_unexpected_recipient(email)
                Mailbox: processed_count = increment_processed_count(processed_count)
        else:
            Mailbox: _ = wait_briefly()

    Mailbox: intake_status = intake_finished(processed_count)
    return intake_status @ Mailbox


_reset_state()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the call intake workflow.")
    parser.add_argument("--llm", default="openai:gpt-4o")
    parser.add_argument("--services", choices=("fake", "live"), default="fake")
    parser.add_argument("--store", dest="store_path")
    parser.add_argument("--table", dest="table_path")
    parser.add_argument("--sheet-id", dest="sheet_id")
    parser.add_argument("--sheet-name", dest="sheet_name")
    parser.add_argument("--table-targets", dest="table_targets")
    parser.add_argument("--certified", dest="certified_senders")
    parser.add_argument("--recipient", dest="intake_recipients")
    parser.add_argument("--send-mode", choices=("draft", "send", "log"), default=None)
    parser.add_argument("--send-limit-per-hour", type=int, default=None)
    parser.add_argument("--poll-seconds", type=float, default=None)
    parser.add_argument("--max-messages", type=int)
    parser.add_argument("--timeout", type=float, default=0.0)
    parser.add_argument("--llm-idle-timeout", type=float)
    parser.add_argument("--no-ui", action="store_true")
    args = parser.parse_args()

    configure_call_intake(
        services=args.services,
        certified_senders=args.certified_senders,
        intake_recipients=args.intake_recipients,
        table_path=args.table_path,
        sheet_id=args.sheet_id,
        sheet_name=args.sheet_name,
        table_targets=args.table_targets,
        send_mode=args.send_mode,
        send_limit_per_hour=args.send_limit_per_hour,
        poll_seconds=args.poll_seconds,
        max_messages=args.max_messages,
    )
    call_intake.configure(
        args.llm,
        execution="sqlite",
        store_path=args.store_path,
        timeout=args.timeout,
        ui=not args.no_ui,
        llm_idle_timeout=args.llm_idle_timeout,
    )
    call_intake()
