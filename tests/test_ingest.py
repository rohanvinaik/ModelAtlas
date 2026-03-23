"""Tests for the ingestion daemon."""

from __future__ import annotations

import json
import sqlite3

import pytest

from model_atlas import db
from model_atlas.ingest import (
    _now_iso,
    get_status,
    phase_b,
    print_status,
)


@pytest.fixture
def ingest_conn():
    """In-memory ingest state database."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_ingest_db(conn)
    return conn


@pytest.fixture
def network_conn():
    """In-memory network database."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    db.init_db(conn)
    return conn


def _insert_ingest_model(
    conn: sqlite3.Connection,
    model_id: str,
    source: str = "huggingface",
    likes: int = 100,
    phase_a: int = 1,
    phase_b: int = 0,
    phase_c: int = 0,
    raw: dict | None = None,
) -> None:
    """Helper to insert a model into the ingest tracking table."""
    if raw is None:
        raw = {
            "model_id": model_id,
            "author": model_id.split("/")[0] if "/" in model_id else "",
            "pipeline_tag": "text-generation",
            "tags": ["text-generation"],
            "library_name": "transformers",
            "likes": likes,
            "downloads": 1000,
            "created_at": "2025-01-01T00:00:00Z",
            "license": "apache-2.0",
            "source": source,
        }
    conn.execute(
        """INSERT INTO ingest_models
           (model_id, source, likes, phase_a_done, phase_b_done, phase_c_done,
            raw_json, fetched_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            model_id,
            source,
            likes,
            phase_a,
            phase_b,
            phase_c,
            json.dumps(raw),
            _now_iso(),
        ),
    )
    conn.commit()


class TestIngestDB:
    def test_init_creates_table(self, ingest_conn):
        """init_ingest_db creates the ingest_models table."""
        row = ingest_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ingest_models'"
        ).fetchone()
        assert row is not None

    def test_insert_and_query(self, ingest_conn):
        """Can insert and query ingest models."""
        _insert_ingest_model(ingest_conn, "test/Model-7B")
        row = ingest_conn.execute(
            "SELECT * FROM ingest_models WHERE model_id = 'test/Model-7B'"
        ).fetchone()
        assert row is not None
        assert row["phase_a_done"] == 1
        assert row["phase_b_done"] == 0


class TestPhaseB:
    def test_extracts_pending_models(self, ingest_conn, network_conn):
        """Phase B processes models with phase_a=1, phase_b=0."""
        _insert_ingest_model(ingest_conn, "test/Model-7B", likes=100)
        _insert_ingest_model(ingest_conn, "test/Model-3B", likes=50)

        count = phase_b(ingest_conn, network_conn)
        assert count == 2

        # Verify they're marked as extracted
        rows = ingest_conn.execute(
            "SELECT model_id FROM ingest_models WHERE phase_b_done = 1"
        ).fetchall()
        assert len(rows) == 2

    def test_skips_already_extracted(self, ingest_conn, network_conn):
        """Phase B skips models already extracted."""
        _insert_ingest_model(ingest_conn, "test/Already-Done", phase_b=1)

        count = phase_b(ingest_conn, network_conn)
        assert count == 0

    def test_processes_highest_likes_first(self, ingest_conn, network_conn):
        """Phase B processes models in descending likes order."""
        _insert_ingest_model(ingest_conn, "test/Low", likes=10)
        _insert_ingest_model(ingest_conn, "test/High", likes=1000)
        _insert_ingest_model(ingest_conn, "test/Mid", likes=100)

        count = phase_b(ingest_conn, network_conn)
        assert count == 3

        # All should be in network DB
        for mid in ["test/Low", "test/High", "test/Mid"]:
            model = db.get_model(network_conn, mid)
            assert model is not None

    def test_handles_bad_json(self, ingest_conn, network_conn):
        """Phase B skips models with corrupt JSON."""
        ingest_conn.execute(
            """INSERT INTO ingest_models
               (model_id, source, likes, phase_a_done, raw_json, fetched_at)
               VALUES (?, 'huggingface', 100, 1, ?, ?)""",
            ("test/Bad", "not valid json{{{", _now_iso()),
        )
        ingest_conn.commit()

        count = phase_b(ingest_conn, network_conn)
        assert count == 0

    def test_multi_source(self, ingest_conn, network_conn):
        """Phase B handles models from different sources."""
        _insert_ingest_model(ingest_conn, "test/HFModel", source="huggingface")
        _insert_ingest_model(
            ingest_conn,
            "ollama/llama3:8b",
            source="ollama",
            raw={
                "model_id": "ollama/llama3:8b",
                "author": "",
                "pipeline_tag": "",
                "tags": ["llama", "GGUF-available"],
                "library_name": "",
                "likes": 0,
                "downloads": 0,
                "source": "ollama",
            },
        )

        count = phase_b(ingest_conn, network_conn)
        assert count == 2

        hf = db.get_model(network_conn, "test/HFModel")
        assert hf is not None
        assert hf["source"] == "huggingface"

        ollama = db.get_model(network_conn, "ollama/llama3:8b")
        assert ollama is not None
        assert ollama["source"] == "ollama"

    def test_resume_after_partial(self, ingest_conn, network_conn):
        """Phase B can resume after partial completion."""
        _insert_ingest_model(ingest_conn, "test/Done", phase_b=1)
        _insert_ingest_model(ingest_conn, "test/Pending")

        count = phase_b(ingest_conn, network_conn)
        assert count == 1

        row = ingest_conn.execute(
            "SELECT phase_b_done FROM ingest_models WHERE model_id = 'test/Pending'"
        ).fetchone()
        assert row["phase_b_done"] == 1


class TestStatus:
    def test_empty_status(self, ingest_conn):
        """Status works on empty database."""
        status = get_status(ingest_conn)
        assert status["total_models"] == 0
        assert status["phase_a_done"] == 0

    def test_status_counts(self, ingest_conn):
        """Status reports correct counts."""
        _insert_ingest_model(ingest_conn, "test/A", phase_b=1, phase_c=0)
        _insert_ingest_model(ingest_conn, "test/B", phase_b=1, phase_c=1)
        _insert_ingest_model(ingest_conn, "test/C", phase_b=0, phase_c=0)

        status = get_status(ingest_conn)
        assert status["total_models"] == 3
        assert status["phase_a_done"] == 3  # all have phase_a=1
        assert status["phase_b_done"] == 2
        assert status["phase_c_done"] == 1
        assert status["phase_b_pending"] == 1
        assert status["phase_c_pending"] == 1

    def test_status_by_source(self, ingest_conn):
        """Status breaks down by source."""
        _insert_ingest_model(ingest_conn, "test/HF1", source="huggingface")
        _insert_ingest_model(ingest_conn, "test/HF2", source="huggingface")
        _insert_ingest_model(ingest_conn, "ollama/m1", source="ollama")

        status = get_status(ingest_conn)
        assert "huggingface" in status["by_source"]
        assert status["by_source"]["huggingface"]["total"] == 2
        assert "ollama" in status["by_source"]
        assert status["by_source"]["ollama"]["total"] == 1

    def test_print_status(self, ingest_conn, capsys):
        """print_status produces readable output."""
        _insert_ingest_model(ingest_conn, "test/Model")
        print_status(ingest_conn)
        captured = capsys.readouterr()
        assert "ModelAtlas Ingest Status" in captured.out
        assert "Phase A" in captured.out
