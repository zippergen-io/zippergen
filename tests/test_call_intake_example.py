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


def _lifelines(module):
    return [module.Mailbox, module.Gatekeeper, module.Extractor, module.Table]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


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


def test_upsert_call_record_updates_existing_row(tmp_path):
    module = _load_call_intake()
    table = tmp_path / "calls.csv"
    module.reset_for_tests(
        fake_inbox=[],
        certified_senders="alice@example.com",
        table_path=table,
        response_log_path=tmp_path / "responses.jsonl",
    )
    email = "From: Alice <alice@example.com>\nSubject: Call\n\nBody"
    first = json.dumps({
        "call_id": "call_demo",
        "type": "project",
        "title": "Old title",
        "deadline": "2026-10-01",
    })
    second = json.dumps({
        "call_id": "call_demo",
        "title": "New title",
        "amount_of_funding": "EUR 1M",
    })

    first_status = module.upsert_call_record.fn(email, module.normalize_call_json.fn(email, first))
    second_status = module.upsert_call_record.fn(email, module.normalize_correction_json.fn(email, second))
    rows = _rows(table)

    assert first_status == "created:call_demo"
    assert second_status == "updated:call_demo"
    assert len(rows) == 1
    assert rows[0]["call_id"] == "call_demo"
    assert rows[0]["title"] == "New title"
    assert rows[0]["deadline"] == "2026-10-01"
    assert rows[0]["amount_of_funding"] == "EUR 1M"
    assert rows[0]["status"] == "corrected"


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
    module.upsert_call_record.fn(
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
