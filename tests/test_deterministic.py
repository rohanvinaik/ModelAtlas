"""Tests for Tier 1 deterministic extraction — architecture fallback chain."""

from __future__ import annotations

from model_atlas.extraction.deterministic import (
    ModelInput,
    _extract_architecture,
    extract,
)


class TestExtractArchitecture:
    def test_config_architectures_lookup(self):
        """Config with known architecture class maps correctly."""
        config = {"architectures": ["BertModel"], "model_type": "bert"}
        pos = _extract_architecture(config)
        assert pos.nodes == ["encoder-only"]
        assert pos.sign == -1

    def test_config_model_type_fallback(self):
        """Unknown architecture class falls back to model_type."""
        config = {"architectures": ["UnknownArch"], "model_type": "t5"}
        pos = _extract_architecture(config)
        assert pos.nodes == ["encoder-decoder"]

    def test_config_mixtral_moe(self):
        """Mixtral maps to mixture-of-experts."""
        config = {"architectures": ["MixtralForCausalLM"], "model_type": "mixtral"}
        pos = _extract_architecture(config)
        assert pos.nodes == ["mixture-of-experts"]
        assert pos.sign == 1

    def test_pipeline_tag_fallback(self):
        """No config → falls back to pipeline_tag heuristic."""
        pos = _extract_architecture(None, pipeline_tag="fill-mask")
        assert pos.nodes == ["encoder-only"]

    def test_pipeline_tag_vision(self):
        """Image classification → vision-transformer."""
        pos = _extract_architecture(None, pipeline_tag="image-classification")
        assert pos.nodes == ["vision-transformer"]

    def test_pipeline_tag_diffusion(self):
        """text-to-image → diffusion."""
        pos = _extract_architecture(None, pipeline_tag="text-to-image")
        assert pos.nodes == ["diffusion"]

    def test_pipeline_tag_translation(self):
        """translation → encoder-decoder."""
        pos = _extract_architecture(None, pipeline_tag="translation")
        assert pos.nodes == ["encoder-decoder"]

    def test_no_config_no_pipeline_returns_unknown(self):
        """No config, no pipeline_tag → unknown (not decoder-only)."""
        pos = _extract_architecture(None)
        assert pos.nodes == ["unknown"]

    def test_empty_config_returns_unknown(self):
        """Empty config dict → unknown."""
        pos = _extract_architecture({})
        assert pos.nodes == ["unknown"]

    def test_config_none_returns_unknown(self):
        """config=None → unknown."""
        pos = _extract_architecture(None, pipeline_tag="", library_name="")
        assert pos.nodes == ["unknown"]

    def test_config_takes_priority_over_pipeline_tag(self):
        """Config architectures take priority over pipeline_tag."""
        config = {"architectures": ["BertModel"]}
        pos = _extract_architecture(config, pipeline_tag="text-to-image")
        assert pos.nodes == ["encoder-only"]

    def test_model_type_takes_priority_over_pipeline_tag(self):
        """model_type fallback takes priority over pipeline_tag."""
        config = {"architectures": ["UnknownArch"], "model_type": "mamba"}
        pos = _extract_architecture(config, pipeline_tag="fill-mask")
        assert pos.nodes == ["mamba", "ssm"]


class TestExtractIntegration:
    def test_extract_with_config(self):
        """Full extract() with config produces correct architecture anchors."""
        inp = ModelInput(
            model_id="google/bert-base",
            config={"architectures": ["BertModel"], "model_type": "bert"},
        )
        result = extract(inp)
        arch_labels = [a.label for a in result.anchors if a.bank == "ARCHITECTURE"]
        assert "encoder-only" in arch_labels

    def test_extract_without_config_uses_pipeline(self):
        """Full extract() without config uses pipeline_tag fallback."""
        inp = ModelInput(
            model_id="test/fill-mask-model",
            pipeline_tag="fill-mask",
            config=None,
        )
        result = extract(inp)
        arch_labels = [a.label for a in result.anchors if a.bank == "ARCHITECTURE"]
        assert "encoder-only" in arch_labels

    def test_extract_no_config_no_pipeline_gives_unknown(self):
        """Full extract() with nothing → unknown architecture."""
        inp = ModelInput(model_id="test/bare-model")
        result = extract(inp)
        arch_labels = [a.label for a in result.anchors if a.bank == "ARCHITECTURE"]
        assert "unknown" in arch_labels
        assert "decoder-only" not in arch_labels
