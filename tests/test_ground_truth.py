"""Tests for C4 ground truth validation."""

from __future__ import annotations

import sqlite3

from model_atlas import db
from model_atlas.ground_truth import validate_against_ground_truth


def _make_network_conn() -> sqlite3.Connection:
    """In-memory network database with test data."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    db.init_db(conn)
    return conn


class TestValidateAgainstGroundTruth:
    def test_synthetic_summary_comparison(self):
        """Validates summary similarity against synthetic ground truth."""
        conn = _make_network_conn()

        # Insert a model with a vibe_summary
        db.insert_model(conn, "test/Model-7B", author="test")
        db.set_metadata(
            conn,
            "test/Model-7B",
            "vibe_summary",
            "A 7B parameter language model for code generation",
            "str",
        )
        conn.commit()

        # Synthetic ground truth — similar but not identical
        gt_summaries = {
            "test/Model-7B": "A 7 billion parameter model designed for code generation tasks",
        }
        gt_parsed: dict[str, dict] = {}

        result = validate_against_ground_truth(conn, gt_summaries, gt_parsed)

        assert result["summary_comparisons"] == 1
        assert result["similarity_mean"] > 0.3  # Should be somewhat similar
        assert result["total_compared"] == 1
        assert len(result["flagged_disagreements"]) == 0  # Not low enough to flag

    def test_no_overlap(self):
        """Returns zeros when there's no overlap between our data and ground truth."""
        conn = _make_network_conn()

        gt_summaries = {"other/Model": "Some summary"}
        gt_parsed: dict[str, dict] = {}

        result = validate_against_ground_truth(conn, gt_summaries, gt_parsed)

        assert result["summary_comparisons"] == 0
        assert result["similarity_mean"] == 0.0
        assert result["total_compared"] == 0

    def test_empty_ground_truth(self):
        """Works with empty ground truth datasets."""
        conn = _make_network_conn()

        result = validate_against_ground_truth(conn, {}, {})

        assert result["total_compared"] == 0
        assert result["similarity_mean"] == 0.0
        assert result["anchor_coverage_mean"] == 0.0

    def test_flags_low_similarity(self):
        """Flags models with very low summary similarity."""
        conn = _make_network_conn()

        db.insert_model(conn, "test/Divergent", author="test")
        db.set_metadata(
            conn, "test/Divergent", "vibe_summary", "AAAA BBBB CCCC DDDD", "str"
        )
        conn.commit()

        gt_summaries = {
            "test/Divergent": "A completely different description of a vision transformer model",
        }

        result = validate_against_ground_truth(conn, gt_summaries, {})

        assert result["summary_comparisons"] == 1
        # Should flag this as low similarity
        assert len(result["flagged_disagreements"]) == 1
        assert result["flagged_disagreements"][0]["type"] == "low_summary_similarity"
