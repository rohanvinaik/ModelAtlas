"""Tests for Phase C export/merge orchestration."""

from __future__ import annotations

import json
import sqlite3

import pytest

from model_atlas import db
from model_atlas.ingest_phase_c import (
    export_c1,
    export_c2,
    export_c3,
    get_phase_c_status,
    merge_c1,
    merge_c2,
    merge_c3,
    print_phase_c_status,
    select_summaries,
)


@pytest.fixture
def network_conn():
    """In-memory network database."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    db.init_db(conn)
    return conn


def _add_model(
    conn: sqlite3.Connection,
    model_id: str,
    author: str = "test",
    source: str = "huggingface",
    likes: int = 100,
    pipeline_tag: str = "text-generation",
) -> None:
    """Insert a model with basic metadata."""
    db.insert_model(conn, model_id, author=author, source=source)
    if likes:
        db.set_metadata(conn, model_id, "likes", str(likes), "int")
    if pipeline_tag:
        db.set_metadata(conn, model_id, "pipeline_tag", pipeline_tag, "str")


def _add_anchor(
    conn: sqlite3.Connection,
    model_id: str,
    label: str,
    bank: str = "CAPABILITY",
    source: str = "extraction",
) -> None:
    """Link a model to an anchor."""
    anchor_id = db.get_or_create_anchor(conn, label, bank, source=source)
    db.link_anchor(conn, model_id, anchor_id)


# ---------------------------------------------------------------------------
# Export C2
# ---------------------------------------------------------------------------


class TestExportC2:
    def test_sharded_round_robin(self, network_conn, tmp_path, monkeypatch):
        """Models are distributed round-robin across shards."""
        monkeypatch.setattr("model_atlas.ingest_phase_c.PHASE_C_WORK_DIR", tmp_path)
        for i in range(5):
            _add_model(network_conn, f"test/model-{i}")

        count = export_c2(network_conn, num_shards=2)
        assert count == 5

        shard_0 = (tmp_path / "shard_0.jsonl").read_text().strip().split("\n")
        shard_1 = (tmp_path / "shard_1.jsonl").read_text().strip().split("\n")
        assert len(shard_0) == 3  # indices 0, 2, 4
        assert len(shard_1) == 2  # indices 1, 3

        # Each line is valid JSON with model_id and prompt
        item = json.loads(shard_0[0])
        assert "model_id" in item
        assert "prompt" in item

    def test_min_likes_filter(self, network_conn, tmp_path, monkeypatch):
        """min_likes filters out low-popularity models."""
        monkeypatch.setattr("model_atlas.ingest_phase_c.PHASE_C_WORK_DIR", tmp_path)
        _add_model(network_conn, "test/popular", likes=500)
        _add_model(network_conn, "test/unpopular", likes=5)

        count = export_c2(network_conn, num_shards=1, min_likes=100)
        assert count == 1

        lines = (tmp_path / "shard_0.jsonl").read_text().strip().split("\n")
        item = json.loads(lines[0])
        assert item["model_id"] == "test/popular"

    def test_skips_models_with_qwen_summary(self, network_conn, tmp_path, monkeypatch):
        """Models that already have qwen_summary are skipped."""
        monkeypatch.setattr("model_atlas.ingest_phase_c.PHASE_C_WORK_DIR", tmp_path)
        _add_model(network_conn, "test/done")
        db.set_metadata(
            network_conn, "test/done", "qwen_summary", "already done", "str"
        )
        _add_model(network_conn, "test/pending")

        count = export_c2(network_conn, num_shards=1)
        assert count == 1

    def test_prompt_includes_config_summary(self, network_conn, tmp_path, monkeypatch):
        """Vibe prompt includes config summary from metadata."""
        monkeypatch.setattr("model_atlas.ingest_phase_c.PHASE_C_WORK_DIR", tmp_path)
        _add_model(network_conn, "test/llama-7b")
        db.set_metadata(network_conn, "test/llama-7b", "model_type", "llama", "str")
        db.set_metadata(network_conn, "test/llama-7b", "num_layers", "32", "int")
        db.set_metadata(network_conn, "test/llama-7b", "hidden_size", "4096", "int")

        export_c2(network_conn, num_shards=1)

        lines = (tmp_path / "shard_0.jsonl").read_text().strip().split("\n")
        item = json.loads(lines[0])
        assert "model_type=llama" in item["prompt"]
        assert "num_layers=32" in item["prompt"]
        assert "hidden_size=4096" in item["prompt"]

    def test_prompt_includes_existing_anchors(
        self, network_conn, tmp_path, monkeypatch
    ):
        """Vibe prompt includes existing anchors."""
        monkeypatch.setattr("model_atlas.ingest_phase_c.PHASE_C_WORK_DIR", tmp_path)
        _add_model(network_conn, "test/model-with-anchors")
        _add_anchor(
            network_conn, "test/model-with-anchors", "decoder-only", bank="ARCHITECTURE"
        )
        _add_anchor(
            network_conn, "test/model-with-anchors", "7B-class", bank="EFFICIENCY"
        )

        export_c2(network_conn, num_shards=1)

        lines = (tmp_path / "shard_0.jsonl").read_text().strip().split("\n")
        item = json.loads(lines[0])
        assert "decoder-only" in item["prompt"]
        assert "7B-class" in item["prompt"]

    def test_prompt_has_no_placeholder_tags(self, network_conn, tmp_path, monkeypatch):
        """Exported prompts must not contain literal placeholder strings."""
        monkeypatch.setattr("model_atlas.ingest_phase_c.PHASE_C_WORK_DIR", tmp_path)
        _add_model(network_conn, "test/model-a")

        export_c2(network_conn, num_shards=1)

        lines = (tmp_path / "shard_0.jsonl").read_text().strip().split("\n")
        item = json.loads(lines[0])
        for placeholder in ["tag1", "tag2", "flag1"]:
            assert placeholder not in item["prompt"]

    def test_zero_models_returns_zero(self, network_conn, tmp_path, monkeypatch):
        """No models → returns 0, no files created."""
        monkeypatch.setattr("model_atlas.ingest_phase_c.PHASE_C_WORK_DIR", tmp_path)
        count = export_c2(network_conn, num_shards=2)
        assert count == 0

    def test_output_includes_valid_anchors(self, network_conn, tmp_path, monkeypatch):
        """Each JSONL item includes valid_anchors from CAPABILITY+DOMAIN banks."""
        monkeypatch.setattr("model_atlas.ingest_phase_c.PHASE_C_WORK_DIR", tmp_path)
        _add_model(network_conn, "test/model-a")
        # Add some dictionary anchors
        db.get_or_create_anchor(network_conn, "reasoning", "CAPABILITY", source="seed")
        db.get_or_create_anchor(network_conn, "medical", "DOMAIN", source="seed")

        export_c2(network_conn, num_shards=1)

        lines = (tmp_path / "shard_0.jsonl").read_text().strip().split("\n")
        item = json.loads(lines[0])
        assert "valid_anchors" in item
        assert "reasoning" in item["valid_anchors"]
        assert "medical" in item["valid_anchors"]

    def test_prompt_includes_candidate_lists(self, network_conn, tmp_path, monkeypatch):
        """Prompt lists CAPABILITY and DOMAIN candidates for selection."""
        monkeypatch.setattr("model_atlas.ingest_phase_c.PHASE_C_WORK_DIR", tmp_path)
        _add_model(network_conn, "test/model-a")
        db.get_or_create_anchor(network_conn, "reasoning", "CAPABILITY", source="seed")
        db.get_or_create_anchor(network_conn, "medical", "DOMAIN", source="seed")

        export_c2(network_conn, num_shards=1)

        lines = (tmp_path / "shard_0.jsonl").read_text().strip().split("\n")
        item = json.loads(lines[0])
        assert "reasoning" in item["prompt"]
        assert "medical" in item["prompt"]

    def test_candidates_exclude_already_assigned(
        self, network_conn, tmp_path, monkeypatch
    ):
        """Already-assigned anchors are excluded from candidate lists."""
        monkeypatch.setattr("model_atlas.ingest_phase_c.PHASE_C_WORK_DIR", tmp_path)
        _add_model(network_conn, "test/model-a")
        # Add "reasoning" to dictionary AND link it to the model
        _add_anchor(network_conn, "test/model-a", "reasoning", bank="CAPABILITY")
        # Add "code-generation" to dictionary only (not linked)
        db.get_or_create_anchor(
            network_conn, "code-generation", "CAPABILITY", source="seed"
        )

        export_c2(network_conn, num_shards=1)

        lines = (tmp_path / "shard_0.jsonl").read_text().strip().split("\n")
        item = json.loads(lines[0])
        # "reasoning" is already assigned, so should NOT be in CAPABILITY candidates section
        # But it will be in the "Already assigned" section
        # The candidate list in the prompt should have code-generation but not reasoning
        prompt = item["prompt"]
        cap_section_start = prompt.index("CAPABILITY anchors")
        domain_section_start = prompt.index("DOMAIN anchors")
        cap_section = prompt[cap_section_start:domain_section_start]
        assert "code-generation" in cap_section
        assert "reasoning" not in cap_section


# ---------------------------------------------------------------------------
# Merge C1
# ---------------------------------------------------------------------------


class TestMergeC1:
    def test_merges_valid_results(self, network_conn, tmp_path):
        """Merges smol_summary from JSONL into metadata."""
        _add_model(network_conn, "test/model-a")

        f = tmp_path / "c1_results.jsonl"
        f.write_text(
            json.dumps({"model_id": "test/model-a", "smol_summary": "A great model"})
            + "\n"
        )

        result = merge_c1(network_conn, [str(f)])
        assert result["merged"] == 1
        assert result["errors"] == 0

        val = network_conn.execute(
            "SELECT value FROM model_metadata WHERE model_id = 'test/model-a' AND key = 'smol_summary'"
        ).fetchone()
        assert val[0] == "A great model"

    def test_skips_errors(self, network_conn, tmp_path):
        """Records with 'error' key are skipped."""
        _add_model(network_conn, "test/model-a")

        f = tmp_path / "c1_results.jsonl"
        f.write_text(
            json.dumps({"model_id": "test/model-a", "error": "timeout"}) + "\n"
        )

        result = merge_c1(network_conn, [str(f)])
        assert result["skipped"] == 1
        assert result["merged"] == 0

    def test_creates_stubs_for_unknown_models(self, network_conn, tmp_path):
        """Unknown model_ids get stub model entries."""
        f = tmp_path / "c1_results.jsonl"
        f.write_text(
            json.dumps({"model_id": "unknown/new-model", "smol_summary": "New one"})
            + "\n"
        )

        result = merge_c1(network_conn, [str(f)])
        assert result["merged"] == 1

        model = network_conn.execute(
            "SELECT source FROM models WHERE model_id = 'unknown/new-model'"
        ).fetchone()
        assert model[0] == "stub"

    def test_idempotent(self, network_conn, tmp_path):
        """Merging same file twice doesn't create duplicates."""
        _add_model(network_conn, "test/model-a")

        f = tmp_path / "c1_results.jsonl"
        f.write_text(
            json.dumps({"model_id": "test/model-a", "smol_summary": "Summary v1"})
            + "\n"
        )

        merge_c1(network_conn, [str(f)])
        merge_c1(network_conn, [str(f)])

        count = network_conn.execute(
            "SELECT COUNT(*) FROM model_metadata WHERE model_id = 'test/model-a' AND key = 'smol_summary'"
        ).fetchone()[0]
        assert count == 1


# ---------------------------------------------------------------------------
# Merge C2
# ---------------------------------------------------------------------------


class TestMergeC2:
    def test_stores_summary_and_links_dictionary_anchors(self, network_conn, tmp_path):
        """Merges qwen_summary + links only anchors that exist in dictionary."""
        _add_model(network_conn, "test/model-a")
        # Pre-seed dictionary anchors
        db.get_or_create_anchor(network_conn, "reasoning", "CAPABILITY", source="seed")
        db.get_or_create_anchor(
            network_conn, "code-generation", "CAPABILITY", source="seed"
        )

        f = tmp_path / "c2_results.jsonl"
        f.write_text(
            json.dumps(
                {
                    "model_id": "test/model-a",
                    "summary": "Fine-tuned for code",
                    "selected_anchors": ["reasoning", "code-generation"],
                }
            )
            + "\n"
        )

        result = merge_c2(network_conn, [str(f)])
        assert result["merged"] == 1
        assert result["anchors_linked"] == 2

        val = network_conn.execute(
            "SELECT value FROM model_metadata WHERE model_id = 'test/model-a' AND key = 'qwen_summary'"
        ).fetchone()
        assert val[0] == "Fine-tuned for code"

        anchors = network_conn.execute(
            """SELECT a.label FROM model_anchors ma
               JOIN anchors a ON ma.anchor_id = a.anchor_id
               WHERE ma.model_id = 'test/model-a' AND ma.confidence = 0.5"""
        ).fetchall()
        labels = {r[0] for r in anchors}
        assert "reasoning" in labels
        assert "code-generation" in labels

    def test_rejects_anchors_not_in_dictionary(self, network_conn, tmp_path):
        """Anchors not in the dictionary are silently dropped — no new anchors created."""
        _add_model(network_conn, "test/model-a")
        # Only seed "reasoning", NOT "invented-tag"
        db.get_or_create_anchor(network_conn, "reasoning", "CAPABILITY", source="seed")

        f = tmp_path / "c2_results.jsonl"
        f.write_text(
            json.dumps(
                {
                    "model_id": "test/model-a",
                    "summary": "A model",
                    "selected_anchors": ["reasoning", "invented-tag"],
                }
            )
            + "\n"
        )

        result = merge_c2(network_conn, [str(f)])
        assert result["anchors_linked"] == 1

        # "invented-tag" should NOT exist in anchors table
        row = network_conn.execute(
            "SELECT 1 FROM anchors WHERE label = 'invented-tag'"
        ).fetchone()
        assert row is None

    def test_caps_anchors_at_5(self, network_conn, tmp_path):
        """Selected anchors are capped at 5."""
        _add_model(network_conn, "test/model-a")
        for label in ["a-cap", "b-cap", "c-cap", "d-cap", "e-cap", "f-cap", "g-cap"]:
            db.get_or_create_anchor(network_conn, label, "CAPABILITY", source="seed")

        f = tmp_path / "c2_results.jsonl"
        f.write_text(
            json.dumps(
                {
                    "model_id": "test/model-a",
                    "summary": "A model",
                    "selected_anchors": [
                        "a-cap",
                        "b-cap",
                        "c-cap",
                        "d-cap",
                        "e-cap",
                        "f-cap",
                        "g-cap",
                    ],
                }
            )
            + "\n"
        )

        merge_c2(network_conn, [str(f)])

        anchors = network_conn.execute(
            """SELECT COUNT(*) FROM model_anchors
               WHERE model_id = 'test/model-a' AND confidence = 0.5"""
        ).fetchone()[0]
        assert anchors <= 5

    def test_skips_errors(self, network_conn, tmp_path):
        """Records with 'error' key are skipped."""
        f = tmp_path / "c2_results.jsonl"
        f.write_text(
            json.dumps({"model_id": "test/model-a", "error": "parse failed"}) + "\n"
        )

        result = merge_c2(network_conn, [str(f)])
        assert result["skipped"] == 1
        assert result["merged"] == 0

    def test_backward_compat_extra_anchors(self, network_conn, tmp_path):
        """Falls back to extra_anchors field for old-format results."""
        _add_model(network_conn, "test/model-a")
        db.get_or_create_anchor(
            network_conn, "code-generation", "CAPABILITY", source="seed"
        )

        f = tmp_path / "c2_results.jsonl"
        f.write_text(
            json.dumps(
                {
                    "model_id": "test/model-a",
                    "summary": "Version 1",
                    "extra_anchors": ["code-generation"],
                }
            )
            + "\n"
        )

        result = merge_c2(network_conn, [str(f)])
        assert result["merged"] == 1
        assert result["anchors_linked"] == 1

    def test_idempotent(self, network_conn, tmp_path):
        """Re-merging same data overwrites cleanly."""
        _add_model(network_conn, "test/model-a")
        db.get_or_create_anchor(
            network_conn, "code-generation", "CAPABILITY", source="seed"
        )

        f = tmp_path / "c2_results.jsonl"
        f.write_text(
            json.dumps(
                {
                    "model_id": "test/model-a",
                    "summary": "Version 1",
                    "selected_anchors": ["code-generation"],
                }
            )
            + "\n"
        )

        merge_c2(network_conn, [str(f)])
        merge_c2(network_conn, [str(f)])

        count = network_conn.execute(
            "SELECT COUNT(*) FROM model_metadata WHERE model_id = 'test/model-a' AND key = 'qwen_summary'"
        ).fetchone()[0]
        assert count == 1


# ---------------------------------------------------------------------------
# Export C3
# ---------------------------------------------------------------------------


class TestExportC3:
    def test_exports_models_needing_quality(self, network_conn, tmp_path, monkeypatch):
        """Exports models with vibe_summary but no quality_score."""
        monkeypatch.setattr("model_atlas.ingest_phase_c.PHASE_C3_WORK_DIR", tmp_path)
        _add_model(network_conn, "test/vibed")
        db.set_metadata(
            network_conn, "test/vibed", "vibe_summary", "A cool model", "str"
        )

        _add_model(network_conn, "test/scored")
        db.set_metadata(
            network_conn, "test/scored", "vibe_summary", "Scored model", "str"
        )
        db.set_metadata(network_conn, "test/scored", "quality_score", "0.8", "float")

        count = export_c3(network_conn, num_shards=1)
        assert count == 1

        lines = (tmp_path / "shard_0.jsonl").read_text().strip().split("\n")
        item = json.loads(lines[0])
        assert item["model_id"] == "test/vibed"
        assert "prompt" in item

    def test_blind_prompt_contains_summary(self, network_conn, tmp_path, monkeypatch):
        """Quality gate prompt includes the summary text."""
        monkeypatch.setattr("model_atlas.ingest_phase_c.PHASE_C3_WORK_DIR", tmp_path)
        _add_model(network_conn, "test/model")
        db.set_metadata(
            network_conn, "test/model", "vibe_summary", "Unique summary text", "str"
        )

        export_c3(network_conn, num_shards=1)

        lines = (tmp_path / "shard_0.jsonl").read_text().strip().split("\n")
        item = json.loads(lines[0])
        assert "Unique summary text" in item["prompt"]

    def test_zero_models_returns_zero(self, network_conn, tmp_path, monkeypatch):
        """No eligible models → returns 0."""
        monkeypatch.setattr("model_atlas.ingest_phase_c.PHASE_C3_WORK_DIR", tmp_path)
        count = export_c3(network_conn, num_shards=2)
        assert count == 0


# ---------------------------------------------------------------------------
# Merge C3
# ---------------------------------------------------------------------------


class TestMergeC3:
    def test_stores_quality_score_and_flags(self, network_conn, tmp_path):
        """Merges quality scores and flags."""
        _add_model(network_conn, "test/model-a")

        f = tmp_path / "c3_results.jsonl"
        f.write_text(
            json.dumps(
                {
                    "model_id": "test/model-a",
                    "quality_score": 0.78,
                    "specificity": 2,
                    "coherence": 3,
                    "artifacts": 2,
                    "flags": ["generic"],
                }
            )
            + "\n"
        )

        result = merge_c3(network_conn, [str(f)])
        assert result["merged"] == 1
        assert result["passed"] == 1
        assert result["failed"] == 0

        val = network_conn.execute(
            "SELECT value FROM model_metadata WHERE model_id = 'test/model-a' AND key = 'quality_score'"
        ).fetchone()
        assert float(val[0]) == pytest.approx(0.78)

        flags = network_conn.execute(
            "SELECT value FROM model_metadata WHERE model_id = 'test/model-a' AND key = 'quality_flags'"
        ).fetchone()
        assert json.loads(flags[0]) == ["generic"]

    def test_pass_fail_counting(self, network_conn, tmp_path):
        """Counts passing and failing models correctly."""
        _add_model(network_conn, "test/good")
        _add_model(network_conn, "test/bad")

        f = tmp_path / "c3_results.jsonl"
        lines = [
            json.dumps(
                {
                    "model_id": "test/good",
                    "quality_score": 0.8,
                    "specificity": 3,
                    "coherence": 2,
                    "artifacts": 2,
                    "flags": [],
                }
            ),
            json.dumps(
                {
                    "model_id": "test/bad",
                    "quality_score": 0.2,
                    "specificity": 0,
                    "coherence": 1,
                    "artifacts": 1,
                    "flags": ["generic", "truncated"],
                }
            ),
        ]
        f.write_text("\n".join(lines) + "\n")

        result = merge_c3(network_conn, [str(f)])
        assert result["merged"] == 2
        assert result["passed"] == 1
        assert result["failed"] == 1


# ---------------------------------------------------------------------------
# Select summaries
# ---------------------------------------------------------------------------


class TestSelectSummaries:
    def test_prefers_smol_over_qwen(self, network_conn):
        """When both exist, smol_summary wins."""
        _add_model(network_conn, "test/model-a")
        db.set_metadata(
            network_conn, "test/model-a", "smol_summary", "smol version", "str"
        )
        db.set_metadata(
            network_conn, "test/model-a", "qwen_summary", "qwen version", "str"
        )

        result = select_summaries(network_conn)
        assert result["selected"] == 1
        assert result["smol"] == 1
        assert result["qwen"] == 0

        val = network_conn.execute(
            "SELECT value FROM model_metadata WHERE model_id = 'test/model-a' AND key = 'vibe_summary'"
        ).fetchone()
        assert val[0] == "smol version"

    def test_falls_back_to_qwen(self, network_conn):
        """When only qwen exists, uses qwen_summary."""
        _add_model(network_conn, "test/model-a")
        db.set_metadata(
            network_conn, "test/model-a", "qwen_summary", "qwen version", "str"
        )

        result = select_summaries(network_conn)
        assert result["selected"] == 1
        assert result["qwen"] == 1

        source = network_conn.execute(
            "SELECT value FROM model_metadata WHERE model_id = 'test/model-a' AND key = 'vibe_summary_source'"
        ).fetchone()
        assert source[0] == "qwen"

    def test_stores_provenance(self, network_conn):
        """vibe_summary_source tracks which summary was chosen."""
        _add_model(network_conn, "test/model-a")
        db.set_metadata(
            network_conn, "test/model-a", "smol_summary", "smol text", "str"
        )

        select_summaries(network_conn)

        source = network_conn.execute(
            "SELECT value FROM model_metadata WHERE model_id = 'test/model-a' AND key = 'vibe_summary_source'"
        ).fetchone()
        assert source[0] == "smol"

    def test_preserves_originals(self, network_conn):
        """Original smol_summary and qwen_summary are preserved."""
        _add_model(network_conn, "test/model-a")
        db.set_metadata(
            network_conn, "test/model-a", "smol_summary", "original smol", "str"
        )
        db.set_metadata(
            network_conn, "test/model-a", "qwen_summary", "original qwen", "str"
        )

        select_summaries(network_conn)

        smol = network_conn.execute(
            "SELECT value FROM model_metadata WHERE model_id = 'test/model-a' AND key = 'smol_summary'"
        ).fetchone()
        assert smol[0] == "original smol"

        qwen = network_conn.execute(
            "SELECT value FROM model_metadata WHERE model_id = 'test/model-a' AND key = 'qwen_summary'"
        ).fetchone()
        assert qwen[0] == "original qwen"

    def test_skips_already_done(self, network_conn):
        """Models with existing vibe_summary are skipped."""
        _add_model(network_conn, "test/model-a")
        db.set_metadata(
            network_conn, "test/model-a", "smol_summary", "smol text", "str"
        )
        db.set_metadata(
            network_conn, "test/model-a", "vibe_summary", "already set", "str"
        )

        result = select_summaries(network_conn)
        assert result["selected"] == 0
        assert result["skipped"] == 1

        # Value unchanged
        val = network_conn.execute(
            "SELECT value FROM model_metadata WHERE model_id = 'test/model-a' AND key = 'vibe_summary'"
        ).fetchone()
        assert val[0] == "already set"


# ---------------------------------------------------------------------------
# Phase C status
# ---------------------------------------------------------------------------


class TestPhaseStatus:
    def test_empty_network(self, network_conn):
        """Status works on empty network DB."""
        status = get_phase_c_status(network_conn)
        assert status["total_models"] == 0
        assert status["smol_summary"] == 0
        assert status["qwen_summary"] == 0
        assert status["quality_score"] == 0
        assert status["vibe_summary"] == 0

    def test_counts_metadata_keys(self, network_conn):
        """Status counts metadata keys correctly."""
        _add_model(network_conn, "test/model-a")
        _add_model(network_conn, "test/model-b")
        _add_model(network_conn, "test/model-c")

        db.set_metadata(network_conn, "test/model-a", "smol_summary", "s1", "str")
        db.set_metadata(network_conn, "test/model-b", "smol_summary", "s2", "str")
        db.set_metadata(network_conn, "test/model-a", "qwen_summary", "q1", "str")
        db.set_metadata(network_conn, "test/model-a", "vibe_summary", "v1", "str")

        status = get_phase_c_status(network_conn)
        assert status["total_models"] == 3
        assert status["smol_summary"] == 2
        assert status["qwen_summary"] == 1
        assert status["vibe_summary"] == 1

    def test_pass_fail_breakdown(self, network_conn):
        """Status reports pass/fail quality breakdown."""
        _add_model(network_conn, "test/good")
        _add_model(network_conn, "test/bad")

        db.set_metadata(network_conn, "test/good", "quality_score", "0.8", "float")
        db.set_metadata(network_conn, "test/bad", "quality_score", "0.3", "float")

        status = get_phase_c_status(network_conn)
        assert status["quality_score"] == 2
        assert status["quality_passing"] == 1
        assert status["quality_failing"] == 1

    def test_print_phase_c_status(self, network_conn, capsys):
        """print_phase_c_status produces readable output."""
        _add_model(network_conn, "test/model")
        db.set_metadata(network_conn, "test/model", "smol_summary", "text", "str")

        print_phase_c_status(network_conn)
        captured = capsys.readouterr()
        assert "Phase C Status" in captured.out
        assert "smol_summary" in captured.out


# ---------------------------------------------------------------------------
# Export C1
# ---------------------------------------------------------------------------


class TestExportC1:
    def test_exports_model_ids(self, network_conn, tmp_path, monkeypatch):
        """Exports model_ids without smol_summary."""
        monkeypatch.setattr("model_atlas.ingest_phase_c.PHASE_C1_WORK_DIR", tmp_path)
        _add_model(network_conn, "test/needs-c1")
        _add_model(network_conn, "test/has-c1")
        db.set_metadata(network_conn, "test/has-c1", "smol_summary", "done", "str")

        count = export_c1(network_conn)
        assert count == 1

        lines = (tmp_path / "models_for_c1.jsonl").read_text().strip().split("\n")
        item = json.loads(lines[0])
        assert item["model_id"] == "test/needs-c1"

    def test_zero_returns_zero(self, network_conn, tmp_path, monkeypatch):
        """All models done → returns 0."""
        monkeypatch.setattr("model_atlas.ingest_phase_c.PHASE_C1_WORK_DIR", tmp_path)
        count = export_c1(network_conn)
        assert count == 0
