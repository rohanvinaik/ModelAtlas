"""Tests for the patch_field audit-logged write primitive."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from model_atlas import db
from model_atlas.admin import (
    CANONICAL_TABLES,
    DEFAULT_AUDIT_LOG_PATH,
    InsertResult,
    PatchError,
    PatchResult,
    _audit_log_path_for,
    ensure_anchor,
    insert_canonical,
    patch_field,
    read_audit_archive,
    read_audit_log,
    rotate_audit_log,
)


@pytest.fixture
def file_conn(tmp_path: Path):
    """File-backed SQLite DB so the audit log path can be derived from it."""
    db_path = tmp_path / "test_network.db"
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    db.init_db(connection)
    db.insert_model(
        connection, "meta-llama/Llama-3.1-8B-Instruct", author="meta-llama"
    )
    db.set_position(
        connection,
        "meta-llama/Llama-3.1-8B-Instruct",
        "ARCHITECTURE",
        0,
        0,
        ["decoder-only"],
    )
    connection.commit()
    yield connection
    connection.close()


@pytest.fixture
def audit_log(tmp_path: Path) -> Path:
    return tmp_path / "patches.jsonl"


# --- Contract validation ---


def test_reason_required(file_conn):
    with pytest.raises(PatchError, match="reason"):
        patch_field(
            "models",
            {"model_id": "meta-llama/Llama-3.1-8B-Instruct"},
            "author",
            "new",
            reason="",
            conn=file_conn,
        )


def test_reason_whitespace_only_rejected(file_conn):
    with pytest.raises(PatchError, match="reason"):
        patch_field(
            "models",
            {"model_id": "meta-llama/Llama-3.1-8B-Instruct"},
            "author",
            "new",
            reason="   \n",
            conn=file_conn,
        )


def test_non_canonical_table_rejected(file_conn):
    # model_metadata is observation/derived (Mode 2), not in CANONICAL_TABLES
    with pytest.raises(PatchError, match="not canonical"):
        patch_field(
            "model_metadata",
            {"model_id": "meta-llama/Llama-3.1-8B-Instruct", "key": "x"},
            "value",
            "v",
            reason="testing",
            conn=file_conn,
        )


def test_unknown_table_rejected(file_conn):
    with pytest.raises(PatchError):
        patch_field(
            "no_such_table",
            {"id": 1},
            "x",
            "v",
            reason="testing",
            conn=file_conn,
        )


def test_unknown_field_rejected(file_conn):
    with pytest.raises(PatchError, match="not a column"):
        patch_field(
            "models",
            {"model_id": "meta-llama/Llama-3.1-8B-Instruct"},
            "no_such_col",
            "v",
            reason="testing",
            conn=file_conn,
        )


def test_field_cannot_be_in_key(file_conn):
    with pytest.raises(PatchError, match="identity column"):
        patch_field(
            "models",
            {"model_id": "meta-llama/Llama-3.1-8B-Instruct"},
            "model_id",
            "other",
            reason="testing",
            conn=file_conn,
        )


def test_no_row_match_rejected(file_conn):
    with pytest.raises(PatchError, match="no row"):
        patch_field(
            "models",
            {"model_id": "missing/model"},
            "author",
            "x",
            reason="testing",
            conn=file_conn,
        )


def test_multi_row_match_rejected(file_conn):
    # Add a second position row so a non-PK lookup matches multiple rows
    db.set_position(
        file_conn,
        "meta-llama/Llama-3.1-8B-Instruct",
        "EFFICIENCY",
        0,
        0,
        ["7B-class"],
    )
    file_conn.commit()
    # Lookup by bank alone matches > 1 row in model_positions across models if
    # we add another model. Set up that situation.
    db.insert_model(file_conn, "Qwen/Qwen2.5-Coder-1.5B", author="Qwen")
    db.set_position(file_conn, "Qwen/Qwen2.5-Coder-1.5B", "ARCHITECTURE", 0, 0)
    file_conn.commit()
    with pytest.raises(PatchError, match="matches 2 rows"):
        patch_field(
            "model_positions",
            {"bank": "ARCHITECTURE"},
            "path_depth",
            1,
            reason="testing",
            conn=file_conn,
        )


def test_empty_key_rejected(file_conn):
    with pytest.raises(PatchError, match="non-empty dict"):
        patch_field(
            "models",
            {},
            "author",
            "x",
            reason="testing",
            conn=file_conn,
        )


# --- Dry-run semantics ---


def test_dry_run_returns_diff_without_writing(file_conn, audit_log):
    result = patch_field(
        "models",
        {"model_id": "meta-llama/Llama-3.1-8B-Instruct"},
        "author",
        "new-author",
        reason="dry run check",
        conn=file_conn,
        audit_log_path=audit_log,
    )
    assert isinstance(result, PatchResult)
    assert result.applied is False
    assert result.unchanged is False
    assert result.old_value == "meta-llama"
    assert result.new_value == "new-author"
    # DB unchanged
    row = file_conn.execute(
        "SELECT author FROM models WHERE model_id = ?",
        ("meta-llama/Llama-3.1-8B-Instruct",),
    ).fetchone()
    assert row["author"] == "meta-llama"
    # No audit log written
    assert not audit_log.exists()


def test_unchanged_value_no_audit_entry(file_conn, audit_log):
    result = patch_field(
        "models",
        {"model_id": "meta-llama/Llama-3.1-8B-Instruct"},
        "author",
        "meta-llama",  # same as current
        reason="should be no-op",
        apply=True,
        conn=file_conn,
        audit_log_path=audit_log,
    )
    assert result.unchanged is True
    assert result.applied is False
    assert not audit_log.exists()


# --- Apply semantics ---


def test_apply_writes_db_and_audit_log(file_conn, audit_log):
    result = patch_field(
        "models",
        {"model_id": "meta-llama/Llama-3.1-8B-Instruct"},
        "author",
        "Meta",
        reason="Confirmed via HF API 2026-05-16",
        apply=True,
        conn=file_conn,
        audit_log_path=audit_log,
    )
    file_conn.commit()

    assert result.applied is True
    assert result.unchanged is False
    row = file_conn.execute(
        "SELECT author FROM models WHERE model_id = ?",
        ("meta-llama/Llama-3.1-8B-Instruct",),
    ).fetchone()
    assert row["author"] == "Meta"

    assert audit_log.exists()
    lines = audit_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["table"] == "models"
    assert entry["field"] == "author"
    assert entry["key"] == {"model_id": "meta-llama/Llama-3.1-8B-Instruct"}
    assert entry["old_value"] == "meta-llama"
    assert entry["new_value"] == "Meta"
    assert entry["reason"] == "Confirmed via HF API 2026-05-16"
    assert entry["ts"].endswith("Z")


def test_apply_appends_multiple_entries(file_conn, audit_log):
    patch_field(
        "models",
        {"model_id": "meta-llama/Llama-3.1-8B-Instruct"},
        "author",
        "Meta",
        reason="first patch",
        apply=True,
        conn=file_conn,
        audit_log_path=audit_log,
    )
    patch_field(
        "models",
        {"model_id": "meta-llama/Llama-3.1-8B-Instruct"},
        "display_name",
        "Llama 3.1 8B Instruct",
        reason="second patch",
        apply=True,
        conn=file_conn,
        audit_log_path=audit_log,
    )
    file_conn.commit()

    entries = read_audit_log(audit_log)
    assert len(entries) == 2
    assert entries[0]["reason"] == "first patch"
    assert entries[1]["reason"] == "second patch"


def test_rollback_leaves_no_audit_entry(file_conn, audit_log):
    """Caller rolls back its transaction; the in-flight patch should not commit.

    The audit log line is appended on apply (file write); rollback affects the
    DB but the log line is already on disk. Verify that rollback leaves DB
    unchanged while documenting the log-ahead-of-DB behavior.
    """
    file_conn.execute("BEGIN")
    patch_field(
        "models",
        {"model_id": "meta-llama/Llama-3.1-8B-Instruct"},
        "author",
        "rolled-back",
        reason="will roll back",
        apply=True,
        conn=file_conn,
        audit_log_path=audit_log,
    )
    file_conn.rollback()

    # DB reverted
    row = file_conn.execute(
        "SELECT author FROM models WHERE model_id = ?",
        ("meta-llama/Llama-3.1-8B-Instruct",),
    ).fetchone()
    assert row["author"] == "meta-llama"
    # Audit log entry was written and remains (a known property: the log can
    # be ahead of the DB, never behind). The reconciler/rollback tooling is
    # responsible for reconciling these.
    entries = read_audit_log(audit_log)
    assert len(entries) == 1
    assert entries[0]["new_value"] == "rolled-back"


# --- Audit log path derivation ---


def test_audit_log_path_uses_env_override(monkeypatch, file_conn, tmp_path):
    custom = tmp_path / "custom_patches.jsonl"
    monkeypatch.setenv("MODEL_ATLAS_PATCHES_PATH", str(custom))
    result = patch_field(
        "models",
        {"model_id": "meta-llama/Llama-3.1-8B-Instruct"},
        "author",
        "via-env",
        reason="env-override",
        apply=True,
        conn=file_conn,
    )
    file_conn.commit()
    assert result.audit_log_path == custom
    assert custom.exists()


def test_audit_log_path_colocated_with_test_db(monkeypatch, file_conn, tmp_path):
    monkeypatch.delenv("MODEL_ATLAS_PATCHES_PATH", raising=False)
    derived = _audit_log_path_for(file_conn)
    assert derived == tmp_path / "patches.jsonl"


def test_audit_log_path_memory_conn_raises_without_override(monkeypatch):
    monkeypatch.delenv("MODEL_ATLAS_PATCHES_PATH", raising=False)
    mem_conn = sqlite3.connect(":memory:")
    mem_conn.row_factory = sqlite3.Row
    db.init_db(mem_conn)
    with pytest.raises(PatchError, match="in-memory"):
        _audit_log_path_for(mem_conn)
    mem_conn.close()


def test_canonical_db_path_uses_repo_data_dir(monkeypatch, file_conn):
    """When the connection's main DB is the canonical network.db, the audit
    log path should resolve to <repo>/data/patches.jsonl regardless of cwd."""
    monkeypatch.delenv("MODEL_ATLAS_PATCHES_PATH", raising=False)

    # Build a fake "canonical" conn by patching NETWORK_DB_PATH to the test DB.
    from model_atlas import admin as admin_mod

    test_db = Path(_audit_log_path_for(file_conn)).parent / "test_network.db"
    monkeypatch.setattr(admin_mod, "NETWORK_DB_PATH", test_db)
    derived = _audit_log_path_for(file_conn)
    assert derived == DEFAULT_AUDIT_LOG_PATH


# --- Coverage of CANONICAL_TABLES set ---


def test_canonical_tables_set_matches_documented():
    # Lock the set so changes are deliberate. Tests must be updated alongside.
    assert CANONICAL_TABLES == frozenset(
        {"models", "model_positions", "model_links", "anchors"}
    )


# --- insert_canonical contract ---


def test_insert_canonical_dry_run(file_conn, audit_log):
    result = insert_canonical(
        "models",
        {"model_id": "new/model", "author": "new", "source": "huggingface", "display_name": "M"},
        reason="Phase A seed",
        conn=file_conn,
        audit_log_path=audit_log,
    )
    assert isinstance(result, InsertResult)
    assert result.inserted is False
    # DB unchanged
    row = file_conn.execute(
        "SELECT 1 FROM models WHERE model_id = ?", ("new/model",)
    ).fetchone()
    assert row is None
    assert not audit_log.exists()


def test_insert_canonical_apply(file_conn, audit_log):
    result = insert_canonical(
        "models",
        {"model_id": "new/model", "author": "new-org", "source": "huggingface", "display_name": "M"},
        reason="Phase A seed from HF API 2026-05-16",
        apply=True,
        conn=file_conn,
        audit_log_path=audit_log,
    )
    file_conn.commit()
    assert result.inserted is True
    row = file_conn.execute(
        "SELECT author FROM models WHERE model_id = ?", ("new/model",)
    ).fetchone()
    assert row["author"] == "new-org"

    entries = read_audit_log(audit_log)
    assert len(entries) == 1
    assert entries[0]["op"] == "insert"
    assert entries[0]["table"] == "models"
    assert entries[0]["row"]["model_id"] == "new/model"
    assert entries[0]["reason"] == "Phase A seed from HF API 2026-05-16"


def test_insert_canonical_duplicate_pk_rejected(file_conn, audit_log):
    # The fixture already inserted meta-llama/Llama-3.1-8B-Instruct
    with pytest.raises(PatchError, match="already exists"):
        insert_canonical(
            "models",
            {"model_id": "meta-llama/Llama-3.1-8B-Instruct", "author": "x"},
            reason="duplicate test",
            apply=True,
            conn=file_conn,
            audit_log_path=audit_log,
        )


def test_insert_canonical_missing_pk_rejected(file_conn):
    with pytest.raises(PatchError, match="primary key"):
        insert_canonical(
            "models",
            {"author": "no-pk"},
            reason="missing pk",
            conn=file_conn,
        )


def test_insert_canonical_unknown_column_rejected(file_conn):
    with pytest.raises(PatchError, match="not on"):
        insert_canonical(
            "models",
            {"model_id": "x/y", "bogus_col": "z"},
            reason="bad col",
            conn=file_conn,
        )


def test_insert_canonical_reason_required(file_conn):
    with pytest.raises(PatchError, match="reason"):
        insert_canonical(
            "models",
            {"model_id": "x/y"},
            reason="",
            conn=file_conn,
        )


def test_insert_canonical_non_canonical_table_rejected(file_conn):
    with pytest.raises(PatchError, match="not canonical"):
        insert_canonical(
            "model_metadata",
            {"model_id": "meta-llama/Llama-3.1-8B-Instruct", "key": "k", "value": "v"},
            reason="bypass attempt",
            conn=file_conn,
        )


# --- Rotation ---


def test_rotate_skips_missing(tmp_path):
    p = tmp_path / "patches.jsonl"
    assert rotate_audit_log(p) is None


def test_rotate_skips_empty(tmp_path):
    p = tmp_path / "patches.jsonl"
    p.write_text("")
    assert rotate_audit_log(p) is None


def test_rotate_skips_below_threshold(tmp_path):
    p = tmp_path / "patches.jsonl"
    p.write_text('{"ts":"x","table":"models","field":"a","key":{},"old_value":null,"new_value":"v","reason":"r"}\n')
    assert rotate_audit_log(p, max_bytes=10_000) is None
    assert p.exists() and p.stat().st_size > 0


def test_rotate_archives_and_truncates(tmp_path):
    p = tmp_path / "patches.jsonl"
    payload = '{"ts":"2026-05-16T00:00:00Z","table":"models","field":"a","key":{"id":1},"old_value":null,"new_value":"v","reason":"r"}\n'
    p.write_text(payload * 50)
    archive = rotate_audit_log(p, max_bytes=100)
    assert archive is not None
    assert archive.name == "patches.archive.jsonl.gz"
    assert archive.exists()
    # Active log truncated
    assert p.exists() and p.stat().st_size == 0
    # Archive readable and contains the lines
    entries = read_audit_archive(archive)
    assert len(entries) == 50
    assert entries[0]["reason"] == "r"


# --- ensure_anchor ---


def test_ensure_anchor_creates_new(file_conn, audit_log):
    aid = ensure_anchor(
        file_conn,
        "new-anchor",
        "ARCHITECTURE",
        source="bootstrap",
        reason="seed for testing",
        audit_log_path=audit_log,
    )
    assert isinstance(aid, int) and aid > 0
    entries = read_audit_log(audit_log)
    assert any(e.get("op") == "insert" and e["table"] == "anchors" for e in entries)


def test_ensure_anchor_returns_existing_no_audit(file_conn, audit_log):
    aid1 = ensure_anchor(
        file_conn,
        "shared-anchor",
        "ARCHITECTURE",
        source="bootstrap",
        reason="initial",
        audit_log_path=audit_log,
    )
    first_count = len(read_audit_log(audit_log))
    aid2 = ensure_anchor(
        file_conn,
        "shared-anchor",
        "ARCHITECTURE",
        source="bootstrap",
        reason="should be a no-op",
        audit_log_path=audit_log,
    )
    assert aid1 == aid2
    assert len(read_audit_log(audit_log)) == first_count  # no new entry


def test_ensure_anchor_bank_mismatch_keeps_original(file_conn, audit_log, caplog):
    aid = ensure_anchor(
        file_conn,
        "drift-test",
        "ARCHITECTURE",
        source="bootstrap",
        reason="initial",
        audit_log_path=audit_log,
    )
    with caplog.at_level("WARNING"):
        aid2 = ensure_anchor(
            file_conn,
            "drift-test",
            "CAPABILITY",  # wrong bank
            source="bootstrap",
            reason="reassign attempt",
            audit_log_path=audit_log,
        )
    assert aid2 == aid
    # Bank unchanged in DB
    row = file_conn.execute(
        "SELECT bank FROM anchors WHERE label = ?", ("drift-test",)
    ).fetchone()
    assert row["bank"] == "ARCHITECTURE"
    assert any("ignoring reassignment" in r.message for r in caplog.records)


def test_ensure_anchor_dry_run_on_new_raises(file_conn, audit_log):
    with pytest.raises(PatchError, match="apply=False"):
        ensure_anchor(
            file_conn,
            "new-dry",
            "ARCHITECTURE",
            source="bootstrap",
            reason="dry run on new",
            apply=False,
            audit_log_path=audit_log,
        )


def test_ensure_anchor_reason_required(file_conn):
    with pytest.raises(PatchError):
        ensure_anchor(
            file_conn,
            "x",
            "ARCHITECTURE",
            source="bootstrap",
            reason="",
        )


def test_ensure_anchor_source_required(file_conn):
    with pytest.raises(PatchError, match="source"):
        ensure_anchor(
            file_conn,
            "x",
            "ARCHITECTURE",
            source="",
            reason="r",
        )


def test_rotate_merges_into_existing_archive(tmp_path):
    p = tmp_path / "patches.jsonl"
    line = '{"ts":"2026-05-16T00:00:00Z","table":"models","field":"a","key":{"id":1},"old_value":null,"new_value":"v","reason":"r1"}\n'
    # First rotation
    p.write_text(line * 20)
    archive = rotate_audit_log(p, max_bytes=50)
    assert archive is not None
    assert read_audit_archive(archive) == [json.loads(line)] * 20
    # Second rotation appends to existing archive
    line2 = '{"ts":"2026-05-17T00:00:00Z","table":"models","field":"a","key":{"id":2},"old_value":null,"new_value":"v","reason":"r2"}\n'
    p.write_text(line2 * 10)
    archive2 = rotate_audit_log(p, max_bytes=50)
    assert archive2 == archive
    merged = read_audit_archive(archive)
    assert len(merged) == 30
    assert merged[0]["reason"] == "r1"
    assert merged[-1]["reason"] == "r2"
