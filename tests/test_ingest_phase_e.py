"""Tests for Phase E web enrichment merge and status."""

from __future__ import annotations

import json
import sqlite3

import pytest

from model_atlas import db
from model_atlas.ingest_phase_e import (
    WEB_EXTRACTION_CONFIDENCE,
    _get_existing_anchor_confidence,
    _merge_one_item,
    _parse_jsonl_line,
    _qc_check_contradiction,
    _validate_anchor,
    merge_phase_e,
    phase_e_status,
)


@pytest.fixture
def network_conn():
    """In-memory network database."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    db.init_db(conn)
    return conn


def _add_model(conn, model_id, author="test"):
    db.insert_model(conn, model_id, author=author, source="huggingface")


def _add_anchor_to_db(conn, label, bank="CAPABILITY"):
    return db.get_or_create_anchor(conn, label, bank, source="bootstrap")


# ---------------------------------------------------------------------------
# _parse_jsonl_line
# ---------------------------------------------------------------------------


class TestParseJsonlLine:
    def test_valid_line(self):
        line = json.dumps({"model_id": "x/y", "banks": {"A": {}}})
        item, status = _parse_jsonl_line(line)
        assert status == "ok"
        assert item["model_id"] == "x/y"

    def test_empty_line(self):
        item, status = _parse_jsonl_line("")
        assert status == "empty"
        assert item is None

    def test_whitespace_line(self):
        item, status = _parse_jsonl_line("   \n")
        assert status == "empty"
        assert item is None

    def test_invalid_json(self):
        item, status = _parse_jsonl_line("{broken json")
        assert status == "error"
        assert item is None

    def test_missing_model_id(self):
        line = json.dumps({"banks": {}})
        item, status = _parse_jsonl_line(line)
        assert status == "error"

    def test_error_field_skips(self):
        line = json.dumps({"model_id": "x/y", "error": "timeout"})
        item, status = _parse_jsonl_line(line)
        assert status == "skip"


# ---------------------------------------------------------------------------
# _validate_anchor / _get_existing_anchor_confidence / _qc_check_contradiction
# ---------------------------------------------------------------------------


class TestAnchorValidation:
    def test_validate_known_anchor(self, network_conn):
        aid = _add_anchor_to_db(network_conn, "code-generation")
        result = _validate_anchor(network_conn, "code-generation")
        assert result == aid

    def test_validate_unknown_anchor(self, network_conn):
        result = _validate_anchor(network_conn, "nonexistent-anchor")
        assert result is None

    def test_get_confidence_unlinked(self, network_conn):
        _add_model(network_conn, "x/model")
        _add_anchor_to_db(network_conn, "chat")
        result = _get_existing_anchor_confidence(network_conn, "x/model", 999)
        assert result is None

    def test_get_confidence_linked(self, network_conn):
        _add_model(network_conn, "x/model")
        aid = _add_anchor_to_db(network_conn, "chat")
        db.link_anchor(network_conn, "x/model", aid, confidence=0.85)
        result = _get_existing_anchor_confidence(network_conn, "x/model", aid)
        assert result == 0.85

    def test_qc_contradiction_high_confidence(self, network_conn):
        _add_model(network_conn, "x/model")
        aid = _add_anchor_to_db(network_conn, "chat")
        db.link_anchor(network_conn, "x/model", aid, confidence=0.9)
        assert _qc_check_contradiction(network_conn, "x/model", aid) is True

    def test_qc_no_contradiction_low_confidence(self, network_conn):
        _add_model(network_conn, "x/model")
        aid = _add_anchor_to_db(network_conn, "chat")
        db.link_anchor(network_conn, "x/model", aid, confidence=0.3)
        assert _qc_check_contradiction(network_conn, "x/model", aid) is False

    def test_qc_no_contradiction_unlinked(self, network_conn):
        _add_model(network_conn, "x/model")
        aid = _add_anchor_to_db(network_conn, "chat")
        assert _qc_check_contradiction(network_conn, "x/model", aid) is False


# ---------------------------------------------------------------------------
# _merge_one_item
# ---------------------------------------------------------------------------


class TestMergeOneItem:
    def test_nonexistent_model_returns_zeros(self, network_conn):
        item = {"model_id": "no/such/model", "banks": {}, "source_urls": []}
        stats = _merge_one_item(network_conn, item)
        assert stats["anchors_linked"] == 0

    def test_links_valid_anchor(self, network_conn):
        _add_model(network_conn, "x/model")
        _add_anchor_to_db(network_conn, "reasoning", "CAPABILITY")
        item = {
            "model_id": "x/model",
            "banks": {
                "CAPABILITY": {
                    "selected_anchors": ["reasoning"],
                    "evidence": {"reasoning": "the model reasons well"},
                    "benchmark_scores": {},
                }
            },
            "source_urls": ["https://example.com"],
            "web_summary": "test summary",
        }
        stats = _merge_one_item(network_conn, item)
        assert stats["anchors_linked"] == 1
        # Verify it's in the DB at the right confidence
        row = network_conn.execute(
            "SELECT confidence FROM model_anchors WHERE model_id = ?", ("x/model",)
        ).fetchone()
        assert row[0] == WEB_EXTRACTION_CONFIDENCE

    def test_skips_invalid_anchor(self, network_conn):
        _add_model(network_conn, "x/model")
        item = {
            "model_id": "x/model",
            "banks": {
                "CAPABILITY": {
                    "selected_anchors": ["totally-fake-anchor"],
                    "evidence": {},
                    "benchmark_scores": {},
                }
            },
            "source_urls": [],
            "web_summary": "",
        }
        stats = _merge_one_item(network_conn, item)
        assert stats["anchors_skipped_invalid"] == 1
        assert stats["anchors_linked"] == 0

    def test_skips_existing_higher_confidence(self, network_conn):
        _add_model(network_conn, "x/model")
        aid = _add_anchor_to_db(network_conn, "chat")
        db.link_anchor(network_conn, "x/model", aid, confidence=0.5)
        item = {
            "model_id": "x/model",
            "banks": {
                "CAPABILITY": {
                    "selected_anchors": ["chat"],
                    "evidence": {},
                    "benchmark_scores": {},
                }
            },
            "source_urls": [],
            "web_summary": "",
        }
        stats = _merge_one_item(network_conn, item)
        assert stats["anchors_skipped_existing"] == 1
        # Confidence unchanged
        row = network_conn.execute(
            "SELECT confidence FROM model_anchors WHERE model_id = ?", ("x/model",)
        ).fetchone()
        assert row[0] == 0.5

    def test_stores_benchmark_scores(self, network_conn):
        _add_model(network_conn, "x/model")
        item = {
            "model_id": "x/model",
            "banks": {
                "QUALITY": {
                    "selected_anchors": [],
                    "evidence": {},
                    "benchmark_scores": {"mmlu": 73.2, "humaneval": 45.1},
                }
            },
            "source_urls": [],
            "web_summary": "",
        }
        stats = _merge_one_item(network_conn, item)
        assert stats["benchmarks_stored"] == 2
        row = network_conn.execute(
            "SELECT value FROM model_metadata WHERE model_id = ? AND key = 'benchmark:mmlu'",
            ("x/model",),
        ).fetchone()
        assert row[0] == "73.2"

    def test_dry_run_writes_nothing(self, network_conn):
        _add_model(network_conn, "x/model")
        _add_anchor_to_db(network_conn, "reasoning")
        item = {
            "model_id": "x/model",
            "banks": {
                "CAPABILITY": {
                    "selected_anchors": ["reasoning"],
                    "evidence": {},
                    "benchmark_scores": {"mmlu": 50.0},
                }
            },
            "source_urls": ["https://example.com"],
            "web_summary": "dry run test",
        }
        stats = _merge_one_item(network_conn, item, dry_run=True)
        assert stats["anchors_linked"] == 1
        assert stats["benchmarks_stored"] == 1
        # But nothing actually written
        row = network_conn.execute(
            "SELECT COUNT(*) FROM model_anchors WHERE model_id = ?", ("x/model",)
        ).fetchone()
        assert row[0] == 0

    def test_stores_web_metadata(self, network_conn):
        _add_model(network_conn, "x/model")
        _add_anchor_to_db(network_conn, "reasoning")
        item = {
            "model_id": "x/model",
            "banks": {
                "CAPABILITY": {
                    "selected_anchors": ["reasoning"],
                    "evidence": {},
                    "benchmark_scores": {},
                }
            },
            "source_urls": ["https://example.com"],
            "web_summary": "model does reasoning",
        }
        _merge_one_item(network_conn, item)
        enriched = network_conn.execute(
            "SELECT value FROM model_metadata WHERE model_id = ? AND key = 'web_enriched'",
            ("x/model",),
        ).fetchone()
        assert enriched[0] == "true"
        summary = network_conn.execute(
            "SELECT value FROM model_metadata WHERE model_id = ? AND key = 'web_summary'",
            ("x/model",),
        ).fetchone()
        assert summary[0] == "model does reasoning"

    def test_rejects_out_of_range_benchmark(self, network_conn):
        _add_model(network_conn, "x/model")
        item = {
            "model_id": "x/model",
            "banks": {
                "QUALITY": {
                    "selected_anchors": [],
                    "evidence": {},
                    "benchmark_scores": {"bad": 150.0, "good": 50.0},
                }
            },
            "source_urls": [],
            "web_summary": "",
        }
        stats = _merge_one_item(network_conn, item)
        assert stats["benchmarks_stored"] == 1  # only "good" stored


# ---------------------------------------------------------------------------
# merge_phase_e (integration)
# ---------------------------------------------------------------------------


class TestMergePhaseE:
    def test_full_merge_from_jsonl(self, network_conn, tmp_path):
        _add_model(network_conn, "x/model")
        _add_anchor_to_db(network_conn, "reasoning")

        jsonl = tmp_path / "results.jsonl"
        record = {
            "model_id": "x/model",
            "banks": {
                "CAPABILITY": {
                    "selected_anchors": ["reasoning"],
                    "evidence": {"reasoning": "quote"},
                    "benchmark_scores": {},
                }
            },
            "source_urls": ["https://example.com"],
            "web_summary": "test",
        }
        jsonl.write_text(json.dumps(record) + "\n")

        result = merge_phase_e(network_conn, [str(jsonl)])
        assert result["merged"] == 1
        assert result["anchors_linked"] == 1
        assert result["errors"] == 0

    def test_dry_run_rolls_back(self, network_conn, tmp_path):
        _add_model(network_conn, "x/model")
        _add_anchor_to_db(network_conn, "reasoning")

        jsonl = tmp_path / "results.jsonl"
        record = {
            "model_id": "x/model",
            "banks": {
                "CAPABILITY": {
                    "selected_anchors": ["reasoning"],
                    "evidence": {},
                    "benchmark_scores": {},
                }
            },
            "source_urls": [],
            "web_summary": "",
        }
        jsonl.write_text(json.dumps(record) + "\n")

        result = merge_phase_e(network_conn, [str(jsonl)], dry_run=True)
        assert result["dry_run"] is True
        assert result["anchors_linked"] == 1
        # Nothing persisted
        row = network_conn.execute(
            "SELECT COUNT(*) FROM model_anchors WHERE model_id = ?", ("x/model",)
        ).fetchone()
        assert row[0] == 0

    def test_skips_error_records(self, network_conn, tmp_path):
        jsonl = tmp_path / "results.jsonl"
        lines = [
            json.dumps({"model_id": "x/a", "error": "timeout"}),
            json.dumps({"model_id": "x/b", "banks": {}, "source_urls": [], "web_summary": ""}),
        ]
        jsonl.write_text("\n".join(lines) + "\n")
        _add_model(network_conn, "x/b")

        result = merge_phase_e(network_conn, [str(jsonl)])
        assert result["skipped"] == 1
        assert result["merged"] == 1

    def test_missing_file_raises(self, network_conn):
        with pytest.raises(FileNotFoundError):
            merge_phase_e(network_conn, ["/nonexistent/path.jsonl"])


# ---------------------------------------------------------------------------
# phase_e_status
# ---------------------------------------------------------------------------


class TestPhaseEStatus:
    def test_empty_db(self, network_conn):
        status = phase_e_status(network_conn)
        assert status["web_enriched_models"] == 0
        assert status["total_models"] == 0
        assert status["coverage_pct"] == 0.0
        assert status["web_anchor_links"] == 0
        assert status["benchmark_scores"] == 0
        assert status["recent_runs"] == []

    def test_with_enriched_data(self, network_conn):
        _add_model(network_conn, "x/model")
        db.set_metadata(network_conn, "x/model", "web_enriched", "true", "str")
        db.set_metadata(network_conn, "x/model", "web_summary", "test", "str")
        db.set_metadata(network_conn, "x/model", "benchmark:mmlu", "73.2", "float")

        status = phase_e_status(network_conn)
        assert status["web_enriched_models"] == 1
        assert status["total_models"] == 1
        assert status["coverage_pct"] == 100.0
        assert status["web_summaries"] == 1
        assert status["benchmark_scores"] == 1
        assert status["benchmark_models"] == 1
