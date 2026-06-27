"""Tests for the JSONL reconciler primitive."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from model_atlas import db
from model_atlas.admin import read_audit_log
from model_atlas.reconciler import (
    ReconcileStats,
    ensure_reconciler_schema,
    reconcile_file,
    reconcile_items,
)


@pytest.fixture
def file_conn(tmp_path: Path):
    db_path = tmp_path / "test_network.db"
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    db.init_db(connection)
    db.insert_model(
        connection, "meta-llama/Llama-3.1-8B-Instruct", author="meta-llama"
    )
    connection.commit()
    yield connection
    connection.close()


@pytest.fixture
def audit_log(tmp_path: Path) -> Path:
    return tmp_path / "patches.jsonl"


def _write_jsonl(path: Path, items: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


# --- Schema bootstrap ---


def test_ensure_schema_creates_table(file_conn):
    ensure_reconciler_schema(file_conn)
    row = file_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='reconciler_processed'"
    ).fetchone()
    assert row is not None


def test_ensure_schema_idempotent(file_conn):
    ensure_reconciler_schema(file_conn)
    ensure_reconciler_schema(file_conn)
    # No error; table still has expected shape
    cols = {row[1] for row in file_conn.execute("PRAGMA table_info(reconciler_processed)")}
    assert {"line_hash", "source_file", "processed_at"} <= cols


# --- Insert path ---


def test_reconcile_inserts_new_row(tmp_path, file_conn, audit_log):
    jsonl = tmp_path / "new_models.jsonl"
    _write_jsonl(jsonl, [
        {
            "op": "upsert",
            "table": "models",
            "key": {"model_id": "Qwen/Qwen2.5-Coder-1.5B"},
            "row": {"author": "Qwen", "source": "huggingface", "display_name": "Qwen2.5-Coder-1.5B"},
            "host": "macpro",
            "source_url": "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B",
            "captured_at": "2026-05-16T10:00:00Z",
        },
    ])

    stats = reconcile_file(jsonl, file_conn, apply=True, audit_log_path=audit_log)
    file_conn.commit()

    assert stats.lines_seen == 1
    assert stats.inserts == 1
    assert stats.patches == 0
    assert stats.errors == []

    row = file_conn.execute(
        "SELECT author FROM models WHERE model_id = ?", ("Qwen/Qwen2.5-Coder-1.5B",)
    ).fetchone()
    assert row["author"] == "Qwen"

    entries = read_audit_log(audit_log)
    assert len(entries) == 1
    assert entries[0]["op"] == "insert"
    assert "host=macpro" in entries[0]["reason"]
    assert "captured_at=2026-05-16T10:00:00Z" in entries[0]["reason"]


# --- Patch path ---


def test_reconcile_patches_existing_row(tmp_path, file_conn, audit_log):
    jsonl = tmp_path / "patch.jsonl"
    _write_jsonl(jsonl, [
        {
            "op": "upsert",
            "table": "models",
            "key": {"model_id": "meta-llama/Llama-3.1-8B-Instruct"},
            "row": {"author": "Meta", "display_name": "Llama 3.1 8B Instruct"},
            "host": "macpro",
            "source_url": "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
            "captured_at": "2026-05-16T10:00:00Z",
        },
    ])

    stats = reconcile_file(jsonl, file_conn, apply=True, audit_log_path=audit_log)
    file_conn.commit()

    assert stats.inserts == 0
    assert stats.patches == 1
    assert stats.unchanged == 0

    row = file_conn.execute(
        "SELECT author, display_name FROM models WHERE model_id = ?",
        ("meta-llama/Llama-3.1-8B-Instruct",),
    ).fetchone()
    assert row["author"] == "Meta"
    assert row["display_name"] == "Llama 3.1 8B Instruct"

    entries = read_audit_log(audit_log)
    # Two patches: author, display_name. Both share the reconciler reason.
    assert len(entries) == 2
    assert {e["field"] for e in entries} == {"author", "display_name"}


def test_reconcile_unchanged_no_patch(tmp_path, file_conn, audit_log):
    jsonl = tmp_path / "unchanged.jsonl"
    _write_jsonl(jsonl, [
        {
            "op": "upsert",
            "table": "models",
            "key": {"model_id": "meta-llama/Llama-3.1-8B-Instruct"},
            "row": {"author": "meta-llama"},  # same as current
            "host": "macpro",
            "source_url": "https://huggingface.co/...",
            "captured_at": "2026-05-16T10:00:00Z",
        },
    ])

    stats = reconcile_file(jsonl, file_conn, apply=True, audit_log_path=audit_log)
    file_conn.commit()

    assert stats.patches == 0
    assert stats.unchanged == 1
    # No audit entries written
    assert not audit_log.exists() or read_audit_log(audit_log) == []


# --- Idempotency ---


def test_replay_same_file_is_noop(tmp_path, file_conn, audit_log):
    jsonl = tmp_path / "data.jsonl"
    _write_jsonl(jsonl, [
        {
            "op": "upsert",
            "table": "models",
            "key": {"model_id": "Qwen/Qwen2.5-Coder-1.5B"},
            "row": {"author": "Qwen"},
            "captured_at": "2026-05-16T10:00:00Z",
        },
    ])
    stats1 = reconcile_file(jsonl, file_conn, apply=True, audit_log_path=audit_log)
    file_conn.commit()
    assert stats1.inserts == 1

    stats2 = reconcile_file(jsonl, file_conn, apply=True, audit_log_path=audit_log)
    file_conn.commit()
    assert stats2.lines_skipped_processed == 1
    assert stats2.inserts == 0
    assert stats2.patches == 0
    # Only the first run's audit entry should exist
    assert len(read_audit_log(audit_log)) == 1


def test_different_captured_at_is_new_hash(tmp_path, file_conn, audit_log):
    """Different captured_at on otherwise-identical content yields a different
    line-hash, but the reconciler should produce no diffs (unchanged)."""
    item1 = {
        "op": "upsert",
        "table": "models",
        "key": {"model_id": "Qwen/Qwen2.5-Coder-1.5B"},
        "row": {"author": "Qwen"},
        "captured_at": "2026-05-16T10:00:00Z",
    }
    item2 = dict(item1, captured_at="2026-05-17T10:00:00Z")
    jsonl = tmp_path / "tick.jsonl"
    _write_jsonl(jsonl, [item1, item2])

    stats = reconcile_file(jsonl, file_conn, apply=True, audit_log_path=audit_log)
    file_conn.commit()
    assert stats.lines_seen == 2
    assert stats.lines_skipped_processed == 0
    assert stats.inserts == 1   # first one creates the row
    assert stats.unchanged == 1  # second sees the row exists, no field diff


# --- Dry-run ---


def test_dry_run_no_writes(tmp_path, file_conn, audit_log):
    jsonl = tmp_path / "plan.jsonl"
    _write_jsonl(jsonl, [
        {
            "op": "upsert",
            "table": "models",
            "key": {"model_id": "Qwen/Qwen2.5-Coder-1.5B"},
            "row": {"author": "Qwen"},
            "captured_at": "2026-05-16T10:00:00Z",
        },
    ])
    stats = reconcile_file(jsonl, file_conn, apply=False, audit_log_path=audit_log)
    assert stats.inserts == 1  # would insert
    # No row written
    assert (
        file_conn.execute(
            "SELECT 1 FROM models WHERE model_id = ?", ("Qwen/Qwen2.5-Coder-1.5B",)
        ).fetchone()
        is None
    )
    # No processed-marking either (replay would re-plan)
    assert (
        file_conn.execute("SELECT COUNT(*) FROM reconciler_processed").fetchone()[0]
        == 0
    )


# --- Validation / error paths ---


def test_non_canonical_table_rejected(tmp_path, file_conn, audit_log):
    jsonl = tmp_path / "bad.jsonl"
    _write_jsonl(jsonl, [
        {
            "op": "upsert",
            "table": "model_metadata",
            "key": {"model_id": "x", "key": "y"},
            "row": {"value": "z"},
        },
    ])
    stats = reconcile_file(jsonl, file_conn, apply=True, audit_log_path=audit_log)
    assert stats.inserts == 0
    assert stats.lines_skipped_malformed == 1
    assert any("not canonical" in e for e in stats.errors)


def test_malformed_json_skipped(tmp_path, file_conn, audit_log):
    jsonl = tmp_path / "broken.jsonl"
    jsonl.write_text("not json at all\n", encoding="utf-8")
    stats = reconcile_file(jsonl, file_conn, apply=True, audit_log_path=audit_log)
    assert stats.lines_skipped_malformed == 1
    assert any("JSON decode" in e for e in stats.errors)


def test_patch_op_against_missing_row_errors(tmp_path, file_conn, audit_log):
    jsonl = tmp_path / "patch_missing.jsonl"
    _write_jsonl(jsonl, [
        {
            "op": "patch",
            "table": "models",
            "key": {"model_id": "missing/model"},
            "row": {"author": "x"},
        },
    ])
    stats = reconcile_file(jsonl, file_conn, apply=True, audit_log_path=audit_log)
    assert stats.errors
    assert any("no row" in e for e in stats.errors)


def test_empty_lines_skipped(tmp_path, file_conn, audit_log):
    jsonl = tmp_path / "blanks.jsonl"
    jsonl.write_text("\n\n\n", encoding="utf-8")
    stats = reconcile_file(jsonl, file_conn, apply=True, audit_log_path=audit_log)
    assert stats.lines_seen == 0


# --- ReconcileStats shape ---


def test_stats_is_a_dataclass():
    s = ReconcileStats(file="x")
    assert s.file == "x"
    assert s.lines_seen == 0
    assert s.errors == []


# --- reconcile_items (in-memory) ---


def test_reconcile_items_inserts(file_conn, audit_log):
    items = [
        {
            "op": "upsert",
            "table": "models",
            "key": {"model_id": "Qwen/Qwen2.5-Coder-1.5B"},
            "row": {"author": "Qwen", "source": "huggingface", "display_name": "Q"},
        }
    ]
    stats = reconcile_items(items, file_conn, apply=True, audit_log_path=audit_log)
    file_conn.commit()
    assert stats.inserts == 1
    assert stats.errors == []
    row = file_conn.execute(
        "SELECT author FROM models WHERE model_id = ?", ("Qwen/Qwen2.5-Coder-1.5B",)
    ).fetchone()
    assert row["author"] == "Qwen"


def test_reconcile_items_no_line_hash_tracking(file_conn, audit_log):
    """Replaying the same items should re-do the work (no file means no dedup).

    For in-memory flows, idempotency is the caller's responsibility — the
    typical pattern is "build items once, apply once."
    """
    items = [
        {
            "op": "upsert",
            "table": "models",
            "key": {"model_id": "Qwen/Qwen2.5-Coder-1.5B"},
            "row": {"author": "Qwen"},
        }
    ]
    reconcile_items(items, file_conn, apply=True, audit_log_path=audit_log)
    file_conn.commit()
    stats2 = reconcile_items(items, file_conn, apply=True, audit_log_path=audit_log)
    file_conn.commit()
    # Second pass: row exists, no field diff → unchanged
    assert stats2.inserts == 0
    assert stats2.unchanged == 1


def test_reconcile_items_uses_source_label(file_conn, audit_log):
    items = [
        {
            "op": "insert",
            "table": "models",
            "key": {"model_id": "Qwen/Qwen2.5-Coder-1.5B"},
            "row": {"author": "Qwen"},
        }
    ]
    stats = reconcile_items(
        items, file_conn, apply=True, audit_log_path=audit_log,
        source_label="extract_and_store",
    )
    file_conn.commit()
    assert stats.file == "extract_and_store"
    from model_atlas.admin import read_audit_log
    entries = read_audit_log(audit_log)
    assert any("extract_and_store" in e["reason"] for e in entries)


def test_reconcile_items_validates_each(file_conn, audit_log):
    items = [
        {"op": "insert", "table": "models", "key": {"model_id": "a/b"}, "row": {"author": "a"}},
        {"op": "insert", "table": "model_metadata", "key": {"x": 1}, "row": {"v": 1}},  # not canonical
    ]
    stats = reconcile_items(items, file_conn, apply=True, audit_log_path=audit_log)
    assert stats.inserts == 1
    assert stats.lines_skipped_malformed == 1
    assert any("not canonical" in e for e in stats.errors)
