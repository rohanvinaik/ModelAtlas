"""Tests for the ingest progress tracking database."""

from __future__ import annotations

import sqlite3

from model_atlas.db_ingest import get_connection, init_db


class TestGetConnection:
    def test_returns_connection(self, tmp_path):
        db_path = str(tmp_path / "test_ingest.db")
        conn = get_connection(db_path)
        assert isinstance(conn, sqlite3.Connection)
        # Row factory should be set
        assert conn.row_factory == sqlite3.Row
        conn.close()

    def test_wal_mode_enabled(self, tmp_path):
        db_path = str(tmp_path / "test_ingest.db")
        conn = get_connection(db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        conn.close()


class TestInitDB:
    def test_creates_ingest_models_table(self, tmp_path):
        db_path = str(tmp_path / "test_ingest.db")
        conn = get_connection(db_path)
        init_db(conn)
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ingest_models'"
        ).fetchone()
        assert row is not None
        conn.close()

    def test_idempotent(self, tmp_path):
        """Calling init_db twice doesn't fail."""
        db_path = str(tmp_path / "test_ingest.db")
        conn = get_connection(db_path)
        init_db(conn)
        init_db(conn)
        count = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='ingest_models'"
        ).fetchone()[0]
        assert count == 1
        conn.close()

    def test_schema_columns(self, tmp_path):
        """Table has all expected columns."""
        db_path = str(tmp_path / "test_ingest.db")
        conn = get_connection(db_path)
        init_db(conn)
        # Insert a row to verify columns exist
        conn.execute(
            """INSERT INTO ingest_models
               (model_id, source, likes, phase_a_done, phase_b_done, phase_c_done,
                phase_c_attempts, raw_json, fetched_at, extracted_at, vibed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "test/Model",
                "huggingface",
                100,
                1,
                0,
                0,
                0,
                "{}",
                "2025-01-01",
                None,
                None,
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM ingest_models WHERE model_id = 'test/Model'"
        ).fetchone()
        assert row["model_id"] == "test/Model"
        assert row["source"] == "huggingface"
        assert row["likes"] == 100
        assert row["phase_a_done"] == 1
        assert row["phase_b_done"] == 0
        assert row["phase_c_done"] == 0
        assert row["phase_c_attempts"] == 0
        assert row["raw_json"] == "{}"
        conn.close()

    def test_defaults(self, tmp_path):
        """Default values are applied."""
        db_path = str(tmp_path / "test_ingest.db")
        conn = get_connection(db_path)
        init_db(conn)
        conn.execute("INSERT INTO ingest_models (model_id) VALUES ('test/Default')")
        conn.commit()
        row = conn.execute(
            "SELECT * FROM ingest_models WHERE model_id = 'test/Default'"
        ).fetchone()
        assert row["source"] == "huggingface"
        assert row["likes"] == 0
        assert row["phase_a_done"] == 0
        assert row["phase_b_done"] == 0
        assert row["phase_c_done"] == 0
        assert row["phase_c_attempts"] == 0
        conn.close()
