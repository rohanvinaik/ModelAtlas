"""Tests for the extraction pipeline orchestrator."""

from __future__ import annotations

from model_atlas import db
from model_atlas.extraction.deterministic import ModelInput
from model_atlas.extraction.pipeline import (
    extract_and_store,
    extract_batch,
    infer_relationships,
)


class TestExtractAndStore:
    def test_basic_model_inserted(self, conn):
        """A model processed through the pipeline should exist in the DB."""
        inp = ModelInput(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            author="meta-llama",
            pipeline_tag="text-generation",
            tags=["text-generation", "instruct"],
            library_name="transformers",
            likes=5000,
            downloads=10_000_000,
            config={"architectures": ["LlamaForCausalLM"]},
        )
        extract_and_store(conn, inp, card_text="")
        conn.commit()

        model = db.get_model(conn, "meta-llama/Llama-3.1-8B-Instruct")
        assert model is not None
        assert model["model_id"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert model["author"] == "meta-llama"

    def test_all_seven_banks_populated(self, conn):
        """All 7 semantic banks should get a position after full extraction."""
        inp = ModelInput(
            model_id="test/Model-7B-Instruct",
            author="test",
            pipeline_tag="text-generation",
            tags=["instruct", "code"],
            library_name="transformers",
            likes=100,
            downloads=50000,
            config={"architectures": ["LlamaForCausalLM"]},
        )
        extract_and_store(conn, inp)
        conn.commit()

        model = db.get_model(conn, "test/Model-7B-Instruct")
        assert model is not None
        for bank in db.BANKS:
            assert bank in model["positions"], f"Missing position for bank {bank}"

    def test_deterministic_anchors_stored(self, conn):
        """Anchors from tier-1 deterministic extraction should be in the DB."""
        inp = ModelInput(
            model_id="test/Model-7B",
            author="test",
            safetensors_info={"parameters": {"F16": 7_000_000_000}},
            likes=5000,
            downloads=10_000_000,
        )
        extract_and_store(conn, inp)
        conn.commit()

        anchors = db.get_anchor_set(conn, "test/Model-7B")
        assert "7B-class" in anchors
        assert "high-downloads" in anchors

    def test_pattern_anchors_stored(self, conn):
        """Anchors from tier-2 pattern extraction should be in the DB."""
        inp = ModelInput(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            author="meta-llama",
            tags=["text-generation"],
        )
        extract_and_store(conn, inp)
        conn.commit()

        anchors = db.get_anchor_set(conn, "meta-llama/Llama-3.1-8B-Instruct")
        assert "instruction-following" in anchors
        assert "Llama-family" in anchors

    def test_metadata_stored(self, conn):
        """Overflow metadata should be stored in the metadata table."""
        inp = ModelInput(
            model_id="test/Model-7B",
            author="test",
            license_str="apache-2.0",
            pipeline_tag="text-generation",
            likes=100,
        )
        extract_and_store(conn, inp)
        conn.commit()

        model = db.get_model(conn, "test/Model-7B")
        assert model is not None
        assert "license" in model["metadata"]
        assert model["metadata"]["license"]["value"] == "apache-2.0"

    def test_lineage_link_created(self, conn):
        """When a base model is detected, a fine_tuned_from link should be stored."""
        # Insert the base model first so the foreign key is valid
        db.insert_model(conn, "meta-llama/Llama-3.1-8B", author="meta-llama")

        inp = ModelInput(
            model_id="user/Llama-3.1-8B-LoRA",
            author="user",
            tags=["base_model:meta-llama/Llama-3.1-8B", "lora"],
        )
        extract_and_store(conn, inp)
        conn.commit()

        model = db.get_model(conn, "user/Llama-3.1-8B-LoRA")
        assert model is not None
        outgoing = model["links"]["outgoing"]
        assert any(
            link["target_id"] == "meta-llama/Llama-3.1-8B"
            and link["relation"] == "fine_tuned_from"
            for link in outgoing
        )

    def test_architecture_position_from_config(self, conn):
        """Architecture bank should reflect the model config."""
        inp = ModelInput(
            model_id="test/MambaModel",
            author="test",
            config={"architectures": ["MambaForCausalLM"]},
        )
        extract_and_store(conn, inp)
        conn.commit()

        model = db.get_model(conn, "test/MambaModel")
        assert model is not None
        arch = model["positions"]["ARCHITECTURE"]
        assert arch["sign"] == 1
        assert arch["depth"] == 2

    def test_efficiency_position(self, conn):
        """Efficiency bank should reflect parameter count."""
        inp = ModelInput(
            model_id="test/SmallModel-1.5B",
            author="test",
        )
        extract_and_store(conn, inp)
        conn.commit()

        model = db.get_model(conn, "test/SmallModel-1.5B")
        assert model is not None
        eff = model["positions"]["EFFICIENCY"]
        assert eff["sign"] == -1  # small model

    def test_vibe_summary_empty_by_default(self, conn):
        """Vibe summary should be empty (delegated to Phase C)."""
        inp = ModelInput(
            model_id="test/Model",
            author="test",
        )
        extract_and_store(conn, inp, card_text="A great model for code.")
        conn.commit()

        model = db.get_model(conn, "test/Model")
        assert model is not None
        # Vibe stub returns "" so no vibe_summary metadata should be stored
        assert "vibe_summary" not in model["metadata"]

    def test_quantization_metadata(self, conn):
        """Quantization level should be detected and stored as metadata."""
        inp = ModelInput(
            model_id="user/Model-Q4_K_M-GGUF",
            author="user",
            tags=[],
            library_name="gguf",
        )
        extract_and_store(conn, inp)
        conn.commit()

        model = db.get_model(conn, "user/Model-Q4_K_M-GGUF")
        assert model is not None
        assert "quantization_level" in model["metadata"]
        assert model["metadata"]["quantization_level"]["value"] == "Q4_K_M"


class TestExtractBatch:
    def test_batch_processes_multiple_models(self, conn):
        """extract_batch should process and store all models in the list."""
        models = [
            {
                "model_id": "test/ModelA-7B",
                "author": "test",
                "tags": ["text-generation"],
                "likes": 10,
                "downloads": 1000,
            },
            {
                "model_id": "test/ModelB-3B",
                "author": "test",
                "tags": ["code"],
                "likes": 5,
                "downloads": 500,
            },
        ]
        count = extract_batch(conn, models)
        conn.commit()

        assert count == 2
        assert db.get_model(conn, "test/ModelA-7B") is not None
        assert db.get_model(conn, "test/ModelB-3B") is not None

    def test_batch_returns_count(self, conn):
        models = [
            {"model_id": "test/M1", "author": "a"},
            {"model_id": "test/M2", "author": "b"},
            {"model_id": "test/M3", "author": "c"},
        ]
        count = extract_batch(conn, models)
        assert count == 3

    def test_batch_empty_list(self, conn):
        count = extract_batch(conn, [])
        assert count == 0

    def test_batch_skips_failures(self, conn):
        """If one model fails extraction, the rest should still process."""
        models = [
            {"model_id": "test/Good", "author": "test"},
            # This one has no model_id key at all — will get "" which should still work
            {"tags": ["text-generation"]},
            {"model_id": "test/AlsoGood", "author": "test"},
        ]
        count = extract_batch(conn, models)
        # All three should process (empty model_id is valid for ModelInput)
        assert count >= 2

    def test_batch_preserves_card_text_key(self, conn):
        """The card_text field should be passed through to extract_and_store."""
        models = [
            {
                "model_id": "test/WithCard",
                "author": "test",
                "card_text": "This is a great model.",
            },
        ]
        count = extract_batch(conn, models)
        assert count == 1
        # Model should exist; card_text is consumed by vibes (which is a stub)
        assert db.get_model(conn, "test/WithCard") is not None


class TestInferRelationships:
    def test_sibling_inference(self, conn):
        """Models sharing a base model should get variant_of links."""
        db.insert_model(conn, "base/Model", author="base")
        db.insert_model(conn, "user/FT-A", author="user")
        db.insert_model(conn, "user/FT-B", author="user")
        db.add_link(conn, "user/FT-A", "base/Model", "fine_tuned_from")
        db.add_link(conn, "user/FT-B", "base/Model", "fine_tuned_from")
        conn.commit()

        count = infer_relationships(conn)
        assert count >= 1

        row = conn.execute(
            "SELECT * FROM model_links WHERE relation='variant_of'"
        ).fetchone()
        assert row is not None

    def test_variant_inference_shared_prefix(self, conn):
        """Models from same author with shared name prefix get variant_of."""
        db.insert_model(conn, "meta/Llama-7B", author="meta")
        db.insert_model(conn, "meta/Llama-13B", author="meta")
        conn.commit()

        count = infer_relationships(conn)
        assert count >= 1

        row = conn.execute(
            "SELECT * FROM model_links WHERE relation='variant_of'"
        ).fetchone()
        assert row is not None

    def test_fingerprint_inference(self, conn):
        """Models with same structural fingerprint get same_family links."""
        db.insert_model(conn, "org/ModelA", author="org")
        db.insert_model(conn, "org2/ModelB", author="org2")
        db.set_metadata(conn, "org/ModelA", "structural_fingerprint", "abc123", "str")
        db.set_metadata(conn, "org2/ModelB", "structural_fingerprint", "abc123", "str")
        conn.commit()

        count = infer_relationships(conn)
        assert count >= 1

        row = conn.execute(
            "SELECT * FROM model_links WHERE relation='same_family'"
        ).fetchone()
        assert row is not None

    def test_no_op_on_single_model(self, conn):
        """Single model should produce no inferred links."""
        db.insert_model(conn, "test/Solo", author="test")
        conn.commit()

        count = infer_relationships(conn)
        assert count == 0

    def test_confidence_variation_stored(self, conn):
        """Pattern anchors should store per-category confidence, not flat 0.8."""
        inp = ModelInput(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            author="meta-llama",
            tags=["text-generation", "base_model:meta-llama/Llama-3.1-8B"],
            library_name="transformers",
        )
        db.insert_model(conn, "meta-llama/Llama-3.1-8B", author="meta-llama")
        extract_and_store(conn, inp)
        conn.commit()

        # Check that not all pattern anchors have confidence=0.8
        rows = conn.execute(
            """SELECT DISTINCT confidence FROM model_anchors
               WHERE model_id = ?""",
            ("meta-llama/Llama-3.1-8B-Instruct",),
        ).fetchall()
        confidences = {row[0] for row in rows}
        # Should have more than just {1.0} — pattern anchors vary
        assert len(confidences) > 1
