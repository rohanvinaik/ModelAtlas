"""Tests for persisting a Phase C vibe result to the network DB.

Regression guard. `_store_vibe_result()` read `result.extra_anchors`, but the
object it is handed is a `VibeOutput` from `VibeExtractor.extract()`, whose
anchor field is `selected_anchors` — so the daemon's Phase C raised
AttributeError the moment it stored a result. The parameter was annotated
`object` with a blanket `# type: ignore`, which hid the mismatch from the type
checker, and nothing exercised the function, which hid it from the suite.
`extra_anchors` is the pre-v0.3 spelling and the dict-based merge paths still
accept it, so it stays honoured for a result object that carries it.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import pytest

from model_atlas import db
from model_atlas.extraction.vibes import VibeOutput
from model_atlas.ingest_vibes import _store_vibe_result


@pytest.fixture
def network_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    db.init_db(conn)
    db.insert_model(conn, "test/model-a", author="test", source="huggingface")
    conn.commit()
    return conn


def _anchors_on(conn: sqlite3.Connection, model_id: str) -> set[str]:
    rows = conn.execute(
        """SELECT a.label FROM model_anchors ma
           JOIN anchors a ON ma.anchor_id = a.anchor_id
           WHERE ma.model_id = ?""",
        (model_id,),
    ).fetchall()
    return {r["label"] for r in rows}


def _summary_of(conn: sqlite3.Connection, model_id: str) -> str | None:
    row = conn.execute(
        "SELECT value FROM model_metadata WHERE model_id = ? AND key = 'vibe_summary'",
        (model_id,),
    ).fetchone()
    return row["value"] if row else None


def test_stores_summary_and_selected_anchors(network_conn):
    """The real extractor output shape must round-trip without raising."""
    result = VibeOutput(summary="A compact Go specialist.", selected_anchors=["code-generation"])
    _store_vibe_result(network_conn, "test/model-a", result)

    assert _summary_of(network_conn, "test/model-a") == "A compact Go specialist."
    assert "code-generation" in _anchors_on(network_conn, "test/model-a")


def test_vibe_output_has_no_extra_anchors_attribute():
    """Pins the mismatch that caused the bug: reading `extra_anchors` off a
    real VibeOutput is an AttributeError, not a missing-but-tolerated field."""
    result = VibeOutput(summary="s", selected_anchors=[])
    assert not hasattr(result, "extra_anchors")
    with pytest.raises(AttributeError):
        _ = result.extra_anchors  # type: ignore[attr-defined]


def test_legacy_extra_anchors_object_is_still_honoured(network_conn):
    """The pre-v0.3 field name, matching what the dict-based merge paths do."""

    @dataclass
    class LegacyVibe:
        summary: str
        selected_anchors: list[str]
        extra_anchors: list[str]

    _store_vibe_result(
        network_conn,
        "test/model-a",
        LegacyVibe(summary="s", selected_anchors=[], extra_anchors=["reasoning"]),  # type: ignore[arg-type]
    )
    assert "reasoning" in _anchors_on(network_conn, "test/model-a")


def test_empty_anchor_labels_are_skipped(network_conn):
    result = VibeOutput(summary="s", selected_anchors=["", "   ", "tool-calling"])
    _store_vibe_result(network_conn, "test/model-a", result)
    assert _anchors_on(network_conn, "test/model-a") == {"tool-calling"}


def test_blank_summary_stores_no_metadata(network_conn):
    _store_vibe_result(network_conn, "test/model-a", VibeOutput(summary="", selected_anchors=[]))
    assert _summary_of(network_conn, "test/model-a") is None
