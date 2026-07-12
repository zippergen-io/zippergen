import csv
import importlib.util
import json
from pathlib import Path

from zippergen.runtime import run


def _load_call_intake():
    path = Path(__file__).parents[1] / "examples" / "call_intake.py"
    spec = importlib.util.spec_from_file_location("call_intake_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_call_intake_email_client():
    path = Path(__file__).parents[1] / "examples" / "call_intake_email_client.py"
    spec = importlib.util.spec_from_file_location("call_intake_email_client_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lifelines(module):
    return [module.Mailbox, module.Gatekeeper, module.Extractor, module.Table]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


class FakeSheetsClient:
    def __init__(self, rows=None):
        self.rows = [dict(row) for row in (rows or [])]
        self.reads = []
        self.writes = []

    def read_rows(self, spreadsheet_id, sheet_name, fields):
        self.reads.append((spreadsheet_id, sheet_name, tuple(fields)))
        return [dict(row) for row in self.rows]

    def write_rows(self, spreadsheet_id, sheet_name, fields, rows):
        self.writes.append((spreadsheet_id, sheet_name, tuple(fields), [dict(row) for row in rows]))
        self.rows = [dict(row) for row in rows]


def test_certified_sender_accepts_exact_and_domain(tmp_path):
    module = _load_call_intake()
    module.reset_for_tests(
        fake_inbox=[],
        certified_senders="alice@example.com,@trusted.example",
        table_path=tmp_path / "calls.csv",
        response_log_path=tmp_path / "responses.jsonl",
    )

    assert module.is_certified_sender.fn("From: Alice <alice@example.com>\n\nHi") is True
    assert module.is_certified_sender.fn("From: Bob <bob@trusted.example>\n\nHi") is True
    assert module.is_certified_sender.fn("From: Eve <eve@other.example>\n\nHi") is False


def test_expected_recipient_matches_to_cc_and_delivery_headers(tmp_path):
    module = _load_call_intake()
    module.reset_for_tests(
        fake_inbox=[],
        certified_senders="alice@example.com",
        intake_recipients="zippergen.sandbox+calls@gmail.com",
        table_path=tmp_path / "calls.csv",
        response_log_path=tmp_path / "responses.jsonl",
    )

    assert module.is_expected_intake_recipient.fn(
        "From: Alice <alice@example.com>\n"
        "To: ZipperGen Calls <zippergen.sandbox+calls@gmail.com>\n\nBody"
    ) is True
    assert module.is_expected_intake_recipient.fn(
        "From: Alice <alice@example.com>\n"
        "To: someone@example.com\n"
        "Cc: zippergen.sandbox+calls@gmail.com\n\nBody"
    ) is True
    assert module.is_expected_intake_recipient.fn(
        "From: Alice <alice@example.com>\n"
        "Delivered-To: zippergen.sandbox+calls@gmail.com\n\nBody"
    ) is True
    assert module.is_expected_intake_recipient.fn(
        "From: Alice <alice@example.com>\n"
        "To: other@example.com\n\nBody"
    ) is False


def test_poll_interval_defaults_to_email_scale_and_can_be_overridden(tmp_path):
    module = _load_call_intake()
    module.configure_call_intake(
        services="fake",
        certified_senders="alice@example.com",
        table_path=tmp_path / "calls.csv",
        response_log_path=tmp_path / "responses.jsonl",
        fake_inbox_path=tmp_path / "missing.json",
        send_mode="log",
        max_messages=1,
    )
    assert module._poll_seconds == 60.0

    module.configure_call_intake(
        services="fake",
        certified_senders="alice@example.com",
        table_path=tmp_path / "calls.csv",
        response_log_path=tmp_path / "responses.jsonl",
        fake_inbox_path=tmp_path / "missing.json",
        send_mode="log",
        max_messages=1,
        poll_seconds=300,
    )
    assert module._poll_seconds == 300.0


def test_mail_present_treats_gmail_poll_failure_as_no_mail(tmp_path):
    module = _load_call_intake()
    module.reset_for_tests(
        fake_inbox=[],
        certified_senders="alice@example.com",
        table_path=tmp_path / "calls.csv",
        response_log_path=tmp_path / "responses.jsonl",
    )

    class FailingEmailClient:
        def count_unread(self):
            raise RuntimeError("temporary dns failure")

    module._email_client = FailingEmailClient()

    assert module.mail_present() is False


def test_pop_pending_email_retries_transient_fetch_failure(tmp_path):
    module = _load_call_intake()
    module.reset_for_tests(
        fake_inbox=[],
        certified_senders="alice@example.com",
        table_path=tmp_path / "calls.csv",
        response_log_path=tmp_path / "responses.jsonl",
    )
    module._poll_seconds = 0

    class FlakyEmailClient:
        def __init__(self):
            self.calls = 0

        def fetch_one_unread(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary gmail failure")
            return {
                "sender": "Alice <alice@example.com>",
                "sender_email": "alice@example.com",
                "to": "zippergen.sandbox+calls@gmail.com",
                "subject": "Call",
                "body": "Deadline 2026-10-15",
                "message_id": "<m1@example.com>",
            }

    client = FlakyEmailClient()
    module._email_client = client

    email = module.pop_pending_email.fn()

    assert client.calls == 2
    assert "From: Alice <alice@example.com>" in email
    assert "Message-ID: <m1@example.com>" in email


def test_normalize_call_json_adds_call_id_and_source_fields(tmp_path):
    module = _load_call_intake()
    email = """From: Alice <alice@example.com>
Subject: ANR call
Message-ID: <m1@example.com>

Deadline is 2026-12-01.
"""
    raw = json.dumps({
        "type": "project",
        "title": "ANR AI systems",
        "funding_organism": "ANR",
        "deadline": "2026-12-01",
        "extra": {"instrument": "PRC"},
    })

    normalized = json.loads(module.normalize_call_json.fn(email, raw))

    assert normalized["call_id"].startswith("call_")
    assert normalized["status"] == "new"
    assert normalized["type"] == "project"
    assert normalized["funding_organism"] == "ANR"
    assert normalized["source_sender"] == "alice@example.com"
    assert normalized["source_subject"] == "ANR call"
    assert normalized["source_message_id"] == "<m1@example.com>"
    assert json.loads(normalized["extra_json"]) == {"extra": {"instrument": "PRC"}}


def test_normalize_call_json_uses_gmail_id_when_message_id_is_missing(tmp_path):
    module = _load_call_intake()
    email = """From: Alice <alice@example.com>
Subject: ANR call
Gmail-ID: gmail-message-1

Deadline is 2026-12-01.
"""
    raw = json.dumps({"call_id": "call_demo", "title": "ANR AI systems"})

    normalized = json.loads(module.normalize_call_json.fn(email, raw))

    assert normalized["source_message_id"] == "gmail-message-1"


def test_insert_call_record_skips_duplicate_without_mutation(tmp_path):
    module = _load_call_intake()
    table = tmp_path / "calls.csv"
    module.reset_for_tests(
        fake_inbox=[],
        certified_senders="alice@example.com",
        table_path=table,
        response_log_path=tmp_path / "responses.jsonl",
    )
    first_email = "From: Alice <alice@example.com>\nSubject: Call\nMessage-ID: <m1@example.com>\n\nBody"
    duplicate_email = "From: Alice <alice@example.com>\nSubject: Call\nMessage-ID: <m2@example.com>\n\nBody"
    first = json.dumps({
        "call_id": "call_demo",
        "type": "project",
        "title": "Old title",
        "deadline": "2026-10-01",
    })
    duplicate = json.dumps({
        "call_id": "call_demo",
        "title": "New title",
        "amount_of_funding": "EUR 1M",
    })

    first_status = module.insert_call_record.fn(first_email, module.normalize_call_json.fn(first_email, first))
    retry_status = module.insert_call_record.fn(first_email, module.normalize_call_json.fn(first_email, first))
    duplicate_status = module.insert_call_record.fn(
        duplicate_email,
        module.normalize_call_json.fn(duplicate_email, duplicate),
    )
    rows = _rows(table)

    assert first_status == "created:call_demo"
    assert retry_status == "created:call_demo"
    assert duplicate_status == "duplicate:call_demo"
    assert len(rows) == 1
    assert rows[0]["call_id"] == "call_demo"
    assert rows[0]["title"] == "Old title"
    assert rows[0]["deadline"] == "2026-10-01"
    assert rows[0]["amount_of_funding"] == ""
    assert rows[0]["status"] == "new"
    assert rows[0]["source_message_id"] == "<m1@example.com>"


def test_insert_call_record_skips_same_source_even_with_changed_call_id(tmp_path):
    module = _load_call_intake()
    table = tmp_path / "calls.csv"
    module.reset_for_tests(
        fake_inbox=[],
        certified_senders="alice@example.com",
        table_path=table,
        response_log_path=tmp_path / "responses.jsonl",
    )
    email = "From: Alice <alice@example.com>\nSubject: Call\nMessage-ID: <m1@example.com>\n\nBody"
    first = json.dumps({"call_id": "call_original", "title": "Original title"})
    retry = json.dumps({"call_id": "call_changed", "title": "Changed title"})

    first_status = module.insert_call_record.fn(email, module.normalize_call_json.fn(email, first))
    retry_status = module.insert_call_record.fn(email, module.normalize_call_json.fn(email, retry))
    rows = _rows(table)

    assert first_status == "created:call_original"
    assert retry_status == "created:call_original"
    assert len(rows) == 1
    assert rows[0]["call_id"] == "call_original"
    assert rows[0]["title"] == "Original title"


def test_insert_call_record_mirrors_to_google_sheet_and_csv(tmp_path):
    module = _load_call_intake()
    table = tmp_path / "calls.csv"
    fake_sheets = FakeSheetsClient()
    module.reset_for_tests(
        fake_inbox=[],
        certified_senders="alice@example.com",
        table_path=table,
        response_log_path=tmp_path / "responses.jsonl",
        sheet_id="sheet-1",
        sheet_name="Calls",
        table_targets="both",
    )
    module._sheets_client = fake_sheets
    email = "From: Alice <alice@example.com>\nSubject: Call\nMessage-ID: <m1@example.com>\n\nBody"
    call = json.dumps({
        "call_id": "call_demo",
        "type": "project",
        "title": "Sheet title",
    })

    status = module.insert_call_record.fn(email, module.normalize_call_json.fn(email, call))

    assert status == "created:call_demo"
    assert _rows(table)[0]["title"] == "Sheet title"
    assert fake_sheets.rows[0]["call_id"] == "call_demo"
    assert fake_sheets.rows[0]["title"] == "Sheet title"
    assert fake_sheets.writes[0][0] == "sheet-1"
    assert fake_sheets.writes[0][1] == "Calls"


def test_insert_call_record_detects_duplicate_from_google_sheet(tmp_path):
    module = _load_call_intake()
    table = tmp_path / "calls.csv"
    fake_sheets = FakeSheetsClient(rows=[{
        "call_id": "call_demo",
        "status": "new",
        "title": "Existing sheet title",
        "source_message_id": "<m1@example.com>",
    }])
    module.reset_for_tests(
        fake_inbox=[],
        certified_senders="alice@example.com",
        table_path=table,
        response_log_path=tmp_path / "responses.jsonl",
        sheet_id="sheet-1",
        sheet_name="Calls",
        table_targets="sheets",
    )
    module._sheets_client = fake_sheets
    email = "From: Alice <alice@example.com>\nSubject: Call\nMessage-ID: <m2@example.com>\n\nBody"
    duplicate = json.dumps({
        "call_id": "call_demo",
        "title": "Changed title",
    })

    status = module.insert_call_record.fn(email, module.normalize_call_json.fn(email, duplicate))

    assert status == "duplicate:call_demo"
    assert fake_sheets.rows[0]["title"] == "Existing sheet title"
    assert fake_sheets.writes == []
    assert not table.exists()


def test_apply_call_correction_updates_existing_row(tmp_path):
    module = _load_call_intake()
    table = tmp_path / "calls.csv"
    module.reset_for_tests(
        fake_inbox=[],
        certified_senders="alice@example.com",
        table_path=table,
        response_log_path=tmp_path / "responses.jsonl",
    )
    original_email = "From: Alice <alice@example.com>\nSubject: Call\nMessage-ID: <m1@example.com>\n\nBody"
    correction_email = "From: Alice <alice@example.com>\nSubject: Re: Call\nMessage-ID: <m2@example.com>\n\nBody"
    first = json.dumps({
        "call_id": "call_demo",
        "type": "project",
        "title": "Old title",
        "deadline": "2026-10-01",
    })
    correction = json.dumps({
        "call_id": "call_demo",
        "title": "New title",
        "amount_of_funding": "EUR 1M",
    })

    first_status = module.insert_call_record.fn(
        original_email,
        module.normalize_call_json.fn(original_email, first),
    )
    correction_status = module.apply_call_correction.fn(
        correction_email,
        module.normalize_correction_json.fn(correction_email, correction),
    )
    rows = _rows(table)

    assert first_status == "created:call_demo"
    assert correction_status == "updated:call_demo"
    assert len(rows) == 1
    assert rows[0]["call_id"] == "call_demo"
    assert rows[0]["title"] == "New title"
    assert rows[0]["deadline"] == "2026-10-01"
    assert rows[0]["amount_of_funding"] == "EUR 1M"
    assert rows[0]["status"] == "corrected"


def test_apply_call_correction_updates_google_sheet(tmp_path):
    module = _load_call_intake()
    fake_sheets = FakeSheetsClient(rows=[{
        "call_id": "call_demo",
        "status": "new",
        "type": "project",
        "title": "Old title",
        "deadline": "2026-10-01",
    }])
    module.reset_for_tests(
        fake_inbox=[],
        certified_senders="alice@example.com",
        table_path=tmp_path / "calls.csv",
        response_log_path=tmp_path / "responses.jsonl",
        sheet_id="sheet-1",
        sheet_name="Calls",
        table_targets="sheets",
    )
    module._sheets_client = fake_sheets
    correction_email = "From: Alice <alice@example.com>\nSubject: Re: Call\nMessage-ID: <m2@example.com>\n\nBody"
    correction = json.dumps({
        "call_id": "call_demo",
        "deadline": "2026-11-01",
        "amount_of_funding": "EUR 1M",
    })

    correction_status = module.apply_call_correction.fn(
        correction_email,
        module.normalize_correction_json.fn(correction_email, correction),
    )

    assert correction_status == "updated:call_demo"
    assert fake_sheets.rows[0]["title"] == "Old title"
    assert fake_sheets.rows[0]["deadline"] == "2026-11-01"
    assert fake_sheets.rows[0]["amount_of_funding"] == "EUR 1M"
    assert fake_sheets.rows[0]["status"] == "corrected"
    assert fake_sheets.writes[0][0] == "sheet-1"


def test_apply_call_correction_missing_does_not_create_row(tmp_path):
    module = _load_call_intake()
    table = tmp_path / "calls.csv"
    module.reset_for_tests(
        fake_inbox=[],
        certified_senders="alice@example.com",
        table_path=table,
        response_log_path=tmp_path / "responses.jsonl",
    )
    correction_email = "From: Alice <alice@example.com>\nSubject: Re: Call\nMessage-ID: <m2@example.com>\n\nBody"
    correction = json.dumps({
        "call_id": "call_missing",
        "title": "Missing call",
    })

    correction_status = module.apply_call_correction.fn(
        correction_email,
        module.normalize_correction_json.fn(correction_email, correction),
    )

    assert correction_status == "missing:call_missing"
    assert not table.exists()


def test_send_rate_limit_delay_uses_recent_send_log(tmp_path):
    module = _load_call_intake()
    response_log = tmp_path / "responses.jsonl"
    module.reset_for_tests(
        fake_inbox=[],
        certified_senders="alice@example.com",
        table_path=tmp_path / "calls.csv",
        response_log_path=response_log,
        send_mode="send",
        send_limit_per_hour=1,
    )
    response_log.write_text(json.dumps({
        "key": "sent-1",
        "mode": "send",
        "sent_at": 100.0,
    }) + "\n")

    assert module._send_rate_limit_delay(now=200.0) == 3500.0
    assert module._send_rate_limit_delay(now=3701.0) == 0.0


def test_send_response_uses_gmail_send_and_records_timestamp(tmp_path):
    module = _load_call_intake()
    response_log = tmp_path / "responses.jsonl"
    module.reset_for_tests(
        fake_inbox=[],
        certified_senders="alice@example.com",
        intake_recipients="zippergen.sandbox+calls@gmail.com",
        table_path=tmp_path / "calls.csv",
        response_log_path=response_log,
        send_mode="send",
        send_limit_per_hour=10,
    )

    class FakeEmailClient:
        def __init__(self):
            self.sent = []

        def send_email(self, meta, subject, body):
            self.sent.append((meta, subject, body))
            return "gmail-sent-1"

    fake_client = FakeEmailClient()
    module._email_client = fake_client
    email = "From: Alice <alice@example.com>\nSubject: ERC call\nMessage-ID: <m1@example.com>\n\nBody"
    call_json = module.normalize_call_json.fn(email, json.dumps({"call_id": "call_erc", "title": "ERC"}))

    status = module.send_call_json_response.fn(email, call_json, "created:call_erc")
    records = [json.loads(line) for line in response_log.read_text().splitlines()]

    assert status == "send:gmail-sent-1"
    assert len(fake_client.sent) == 1
    assert fake_client.sent[0][1] == "Extracted call JSON: call_erc"
    assert fake_client.sent[0][0]["sender_email"] == "alice@example.com"
    assert "reply to zippergen.sandbox+calls@gmail.com" in fake_client.sent[0][2]
    assert records[0]["mode"] == "send"
    assert records[0]["recipient"] == "alice@example.com"
    assert records[0]["external_id"] == "gmail-sent-1"
    assert records[0]["source_message_key"] == "<m1@example.com>"
    assert isinstance(records[0]["sent_at"], float)


def test_send_response_skips_same_source_message_even_with_changed_call_id(tmp_path):
    module = _load_call_intake()
    response_log = tmp_path / "responses.jsonl"
    module.reset_for_tests(
        fake_inbox=[],
        certified_senders="alice@example.com",
        table_path=tmp_path / "calls.csv",
        response_log_path=response_log,
        send_mode="send",
        send_limit_per_hour=10,
    )
    response_log.write_text(json.dumps({
        "key": "old-key",
        "mode": "send",
        "source_message_key": "<m1@example.com>",
        "sent_at": 100.0,
    }) + "\n")

    class FakeEmailClient:
        def __init__(self):
            self.sent = []

        def send_email(self, meta, subject, body):
            self.sent.append((meta, subject, body))
            return "gmail-sent-2"

    fake_client = FakeEmailClient()
    module._email_client = fake_client
    email = "From: Alice <alice@example.com>\nSubject: ERC call\nMessage-ID: <m1@example.com>\n\nBody"
    call_json = module.normalize_call_json.fn(email, json.dumps({"call_id": "call_changed", "title": "ERC"}))

    status = module.send_call_json_response.fn(email, call_json, "created:call_changed")

    assert status == "already_recorded:call_changed"
    assert fake_client.sent == []
    assert len(response_log.read_text().splitlines()) == 1


def test_send_response_rejects_missing_sender_address(tmp_path):
    module = _load_call_intake()
    response_log = tmp_path / "responses.jsonl"
    module.reset_for_tests(
        fake_inbox=[],
        certified_senders="alice@example.com",
        table_path=tmp_path / "calls.csv",
        response_log_path=response_log,
        send_mode="send",
    )

    class FakeEmailClient:
        def send_email(self, meta, subject, body):
            raise AssertionError("send_email should not be called")

    module._email_client = FakeEmailClient()
    email = "From: No Address\nSubject: ERC call\nMessage-ID: <m1@example.com>\n\nBody"
    call_json = module.normalize_call_json.fn(email, json.dumps({"call_id": "call_erc", "title": "ERC"}))

    try:
        module.send_call_json_response.fn(email, call_json, "created:call_erc")
    except ValueError as exc:
        assert "sender email address is not valid" in str(exc)
    else:
        raise AssertionError("Expected invalid recipient to be rejected")
    assert not response_log.exists()


def test_gmail_reply_message_requires_sender_recipient_match(monkeypatch):
    client = _load_call_intake_email_client()
    monkeypatch.setenv("ZIPPERGEN_CALL_INTAKE_RECIPIENTS", "zippergen.sandbox+calls@gmail.com")

    msg = client._message_for_reply(
        {"sender": "Alice <alice@example.com>", "sender_email": "alice@example.com"},
        "Extracted call JSON: call_erc",
        "Body",
    )
    assert msg["To"] == "alice@example.com"
    assert msg["Reply-To"] == "zippergen.sandbox+calls@gmail.com"

    try:
        client._message_for_reply(
            {"sender": "Alice <alice@example.com>", "sender_email": "bob@example.com"},
            "Extracted call JSON: call_erc",
            "Body",
        )
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("Expected mismatched recipient to be rejected")


def test_call_intake_skips_llm_for_uncertified_sender(tmp_path):
    module = _load_call_intake()
    module.reset_for_tests(
        fake_inbox=["From: Eve <eve@example.com>\nSubject: Grant\n\nDeadline tomorrow"],
        certified_senders="alice@example.com",
        table_path=tmp_path / "calls.csv",
        response_log_path=tmp_path / "responses.jsonl",
    )

    def backend(action, inputs):
        raise AssertionError(f"LLM should not be called for {action.name}")

    result = run(module.call_intake, _lifelines(module), {}, llm_backend=backend, timeout=5)

    assert result == "processed:1"
    assert not (tmp_path / "calls.csv").exists()


def test_call_intake_skips_llm_for_unexpected_recipient(tmp_path):
    module = _load_call_intake()
    response_log = tmp_path / "responses.jsonl"
    module.reset_for_tests(
        fake_inbox=[
            "From: Alice <alice@example.com>\n"
            "To: other@example.com\n"
            "Subject: Grant\n\nDeadline tomorrow"
        ],
        certified_senders="alice@example.com",
        intake_recipients="zippergen.sandbox+calls@gmail.com",
        table_path=tmp_path / "calls.csv",
        response_log_path=response_log,
    )

    def backend(action, inputs):
        raise AssertionError(f"LLM should not be called for {action.name}")

    result = run(module.call_intake, _lifelines(module), {}, llm_backend=backend, timeout=5)

    assert result == "processed:1"
    assert not (tmp_path / "calls.csv").exists()
    assert not response_log.exists()


def test_call_intake_records_certified_call_and_response(tmp_path):
    module = _load_call_intake()
    table = tmp_path / "calls.csv"
    response_log = tmp_path / "responses.jsonl"
    module.reset_for_tests(
        fake_inbox=["From: Alice <alice@example.com>\nSubject: ERC call\nMessage-ID: <m1@example.com>\n\nDeadline 2026-10-15"],
        certified_senders="alice@example.com",
        table_path=table,
        response_log_path=response_log,
    )

    def backend(action, inputs):
        if action.name == "classify_intake":
            return {"intake_kind": "call"}
        if action.name == "extract_call_json":
            return {
                "call_json": json.dumps({
                    "call_id": "call_erc",
                    "type": "project",
                    "title": "ERC Starting Grant",
                    "funding_organism": "ERC",
                    "deadline": "2026-10-15",
                })
            }
        raise AssertionError(action.name)

    result = run(module.call_intake, _lifelines(module), {}, llm_backend=backend, timeout=5)
    rows = _rows(table)
    response_lines = response_log.read_text().splitlines()

    assert result == "processed:1"
    assert rows[0]["call_id"] == "call_erc"
    assert rows[0]["title"] == "ERC Starting Grant"
    assert rows[0]["funding_organism"] == "ERC"
    assert len(response_lines) == 1
    assert "call_erc" in response_lines[0]


def test_call_intake_reports_duplicate_without_mutating_table(tmp_path):
    module = _load_call_intake()
    table = tmp_path / "calls.csv"
    response_log = tmp_path / "responses.jsonl"
    module.reset_for_tests(
        fake_inbox=[
            "From: Alice <alice@example.com>\nSubject: Duplicate call\nMessage-ID: <m2@example.com>\n\nSecond copy"
        ],
        certified_senders="alice@example.com",
        table_path=table,
        response_log_path=response_log,
    )
    existing_email = (
        "From: Alice <alice@example.com>\n"
        "Subject: Original call\n"
        "Message-ID: <m1@example.com>\n\nBody"
    )
    existing = json.dumps({
        "call_id": "call_demo",
        "type": "project",
        "title": "Original title",
        "deadline": "2026-10-01",
    })
    module.insert_call_record.fn(
        existing_email,
        module.normalize_call_json.fn(existing_email, existing),
    )

    def backend(action, inputs):
        if action.name == "classify_intake":
            return {"intake_kind": "call"}
        if action.name == "extract_call_json":
            return {"call_json": json.dumps({"call_id": "call_demo", "title": "Changed title"})}
        raise AssertionError(action.name)

    result = run(module.call_intake, _lifelines(module), {}, llm_backend=backend, timeout=5)
    rows = _rows(table)
    response = json.loads(response_log.read_text().splitlines()[0])

    assert result == "processed:1"
    assert len(rows) == 1
    assert rows[0]["call_id"] == "call_demo"
    assert rows[0]["title"] == "Original title"
    assert response["purpose"] == "duplicate"
    assert response["subject"] == "Call already recorded: call_demo"
    assert "already exists" in response["body"]


def test_call_intake_correction_updates_existing_row(tmp_path):
    module = _load_call_intake()
    table = tmp_path / "calls.csv"
    response_log = tmp_path / "responses.jsonl"
    module.reset_for_tests(
        fake_inbox=[
            "From: Alice <alice@example.com>\nSubject: Re: Extracted call JSON: call_demo\nMessage-ID: <m2@example.com>\n\nDeadline should be 2026-11-01"
        ],
        certified_senders="alice@example.com",
        table_path=table,
        response_log_path=response_log,
    )
    existing = json.dumps({
        "call_id": "call_demo",
        "type": "project",
        "title": "Demo call",
        "deadline": "2026-10-01",
    })
    module.insert_call_record.fn(
        "From: Alice <alice@example.com>\nSubject: Original\n\nBody",
        module.normalize_call_json.fn("From: Alice <alice@example.com>\nSubject: Original\n\nBody", existing),
    )

    def backend(action, inputs):
        if action.name == "classify_intake":
            return {"intake_kind": "correction"}
        if action.name == "extract_correction_json":
            return {"call_json": json.dumps({"call_id": "call_demo", "deadline": "2026-11-01"})}
        raise AssertionError(action.name)

    result = run(module.call_intake, _lifelines(module), {}, llm_backend=backend, timeout=5)
    rows = _rows(table)

    assert result == "processed:1"
    assert len(rows) == 1
    assert rows[0]["call_id"] == "call_demo"
    assert rows[0]["title"] == "Demo call"
    assert rows[0]["deadline"] == "2026-11-01"
    assert rows[0]["status"] == "corrected"


def test_call_intake_reports_missing_correction_without_creating_row(tmp_path):
    module = _load_call_intake()
    table = tmp_path / "calls.csv"
    response_log = tmp_path / "responses.jsonl"
    module.reset_for_tests(
        fake_inbox=[
            "From: Alice <alice@example.com>\nSubject: Re: Extracted call JSON: call_missing\nMessage-ID: <m2@example.com>\n\nPlease update the deadline"
        ],
        certified_senders="alice@example.com",
        table_path=table,
        response_log_path=response_log,
    )

    def backend(action, inputs):
        if action.name == "classify_intake":
            return {"intake_kind": "correction"}
        if action.name == "extract_correction_json":
            return {"call_json": json.dumps({"call_id": "call_missing", "deadline": "2026-11-01"})}
        raise AssertionError(action.name)

    result = run(module.call_intake, _lifelines(module), {}, llm_backend=backend, timeout=5)
    response = json.loads(response_log.read_text().splitlines()[0])

    assert result == "processed:1"
    assert not table.exists()
    assert response["purpose"] == "missing_correction"
    assert response["subject"] == "Call not found: call_missing"
    assert "could not find" in response["body"]
