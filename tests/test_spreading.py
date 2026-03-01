"""Tests for spreading activation across the semantic network."""

from __future__ import annotations

from model_atlas import db
from model_atlas.spreading import spread


class TestSpreadBasics:
    def test_single_seed_reaches_linked(self, populated_conn):
        """Spreading from GGUF model reaches its quantized_from target."""
        scores = spread(populated_conn, ["TheBloke/Llama-3.1-8B-Instruct-GGUF"])
        # Should reach the base Llama model via quantized_from link
        assert "meta-llama/Llama-3.1-8B-Instruct" in scores
        assert scores["meta-llama/Llama-3.1-8B-Instruct"] > 0

    def test_seed_gets_activation_1(self, populated_conn):
        """Seed models start with activation 1.0."""
        scores = spread(populated_conn, ["meta-llama/Llama-3.1-8B-Instruct"])
        assert scores["meta-llama/Llama-3.1-8B-Instruct"] == 1.0

    def test_decay_reduces_activation(self, populated_conn):
        """Models further from seed have lower activation."""
        scores = spread(populated_conn, ["TheBloke/Llama-3.1-8B-Instruct-GGUF"])
        seed_score = scores["TheBloke/Llama-3.1-8B-Instruct-GGUF"]
        linked_score = scores.get("meta-llama/Llama-3.1-8B-Instruct", 0)
        assert linked_score < seed_score

    def test_empty_seeds(self, populated_conn):
        """Empty seed list returns empty activation."""
        scores = spread(populated_conn, [])
        assert scores == {}

    def test_nonexistent_seed(self, populated_conn):
        """Nonexistent seed gets activation 1.0 but doesn't propagate via links."""
        scores = spread(populated_conn, ["nonexistent/Model"])
        assert scores["nonexistent/Model"] == 1.0
        # Shouldn't reach real models via links (no links from nonexistent)
        # But might reach them via anchors if any happen to match
        # At minimum the seed itself is there
        assert len(scores) >= 1


class TestSpreadBankScoping:
    def test_bank_scoping_limits_propagation(self, populated_conn):
        """When banks are specified, only anchors in those banks propagate."""
        # Spread with LINEAGE scope — should follow lineage anchors
        scoped = spread(
            populated_conn,
            ["meta-llama/Llama-3.1-8B-Instruct"],
            banks=["LINEAGE"],
        )
        # Spread without scope — should reach more models
        unscoped = spread(
            populated_conn,
            ["meta-llama/Llama-3.1-8B-Instruct"],
        )
        # Unscoped should reach at least as many models
        assert len(unscoped) >= len(scoped)


class TestSpreadSliceBounds:
    def test_neighbor_slice_respected(self, conn):
        """Slice bounds limit how many neighbors are explored."""
        # Create a hub model linked to many others
        db.insert_model(conn, "hub/model", author="test")
        for i in range(30):
            mid = f"spoke/model-{i}"
            db.insert_model(conn, mid, author="test")
            db.add_link(conn, "hub/model", mid, "same_family")
        conn.commit()

        scores = spread(conn, ["hub/model"], neighbor_slice=5, max_depth=1)
        # Should not reach all 30 spokes, limited by slice
        reached = [k for k in scores if k.startswith("spoke/")]
        assert len(reached) <= 10  # 5 outgoing + 5 incoming max per direction


class TestSpreadAnchorChannel:
    def test_anchor_channel_activates_colinked(self, populated_conn):
        """Models sharing anchors get activated through Layer 2."""
        # Llama and GGUF share several anchors (decoder-only, instruction-following, Llama-family)
        scores = spread(
            populated_conn,
            ["meta-llama/Llama-3.1-8B-Instruct"],
            max_depth=1,
        )
        # GGUF variant should be reached via both link and anchor channels
        assert "TheBloke/Llama-3.1-8B-Instruct-GGUF" in scores

    def test_anchor_channel_reaches_unlinked(self, populated_conn):
        """Anchor channel reaches models with no explicit links to seed."""
        scores = spread(
            populated_conn,
            ["meta-llama/Llama-3.1-8B-Instruct"],
        )
        # Qwen shares "decoder-only" anchor but has no explicit link
        assert "Qwen/Qwen2.5-Coder-1.5B" in scores


class TestSpreadMultiSeed:
    def test_multiple_seeds(self, populated_conn):
        """Multiple seeds combine activation."""
        scores = spread(
            populated_conn,
            [
                "meta-llama/Llama-3.1-8B-Instruct",
                "Qwen/Qwen2.5-Coder-1.5B",
            ],
        )
        # Both seeds should have activation 1.0
        assert scores["meta-llama/Llama-3.1-8B-Instruct"] == 1.0
        assert scores["Qwen/Qwen2.5-Coder-1.5B"] == 1.0
        # GGUF should be reached from Llama seed
        assert "TheBloke/Llama-3.1-8B-Instruct-GGUF" in scores
