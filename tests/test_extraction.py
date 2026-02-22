"""Tests for the extraction pipeline."""

from __future__ import annotations

from hf_model_search.extraction.deterministic import ModelInput, extract as extract_det
from hf_model_search.extraction.patterns import extract as extract_pat
from hf_model_search.extraction.vibes import extract_vibe_summary


class TestDeterministic:
    def test_architecture_llama(self):
        inp = ModelInput(
            model_id="meta-llama/Llama-3.1-8B",
            config={"architectures": ["LlamaForCausalLM"]},
        )
        result = extract_det(inp)
        assert result.architecture.sign == 0
        assert result.architecture.depth == 0
        assert "decoder-only" in result.architecture.nodes

    def test_architecture_mamba(self):
        inp = ModelInput(
            model_id="state-spaces/mamba-2.8b",
            config={"architectures": ["MambaForCausalLM"]},
        )
        result = extract_det(inp)
        assert result.architecture.sign == 1
        assert result.architecture.depth == 2

    def test_efficiency_from_name(self):
        inp = ModelInput(model_id="test/Model-1.5B")
        result = extract_det(inp)
        assert result.efficiency.sign == -1
        anchors = [a[0] for a in result.anchors]
        assert "3B-class" in anchors  # 1.5B falls in the 1.5-5B range
        assert "consumer-GPU-viable" in anchors

    def test_efficiency_from_safetensors(self):
        inp = ModelInput(
            model_id="test/Model",
            safetensors_info={"parameters": {"F16": 7_000_000_000}},
        )
        result = extract_det(inp)
        assert result.efficiency.sign == 0  # 7B = zero state
        anchors = [a[0] for a in result.anchors]
        assert "7B-class" in anchors

    def test_quality_popular(self):
        inp = ModelInput(
            model_id="test/Popular",
            likes=5000,
            downloads=10_000_000,
        )
        result = extract_det(inp)
        assert result.quality.sign == 1
        anchors = [a[0] for a in result.anchors]
        assert "high-downloads" in anchors
        assert "community-favorite" in anchors

    def test_metadata_collected(self):
        inp = ModelInput(
            model_id="test/Model-7B",
            license_str="apache-2.0",
            pipeline_tag="text-generation",
            likes=100,
        )
        result = extract_det(inp)
        assert result.metadata["license"] == ("apache-2.0", "str")
        assert result.metadata["pipeline_tag"] == ("text-generation", "str")


class TestPatterns:
    def test_capability_instruct(self):
        result = extract_pat(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            tags=["text-generation"],
        )
        anchors = [a[0] for a in result.anchors]
        assert "instruction-following" in anchors
        assert result.capability.sign == 1

    def test_capability_code(self):
        result = extract_pat(
            model_id="Qwen/Qwen2.5-Coder-7B",
            tags=["code"],
        )
        anchors = [a[0] for a in result.anchors]
        assert "code-generation" in anchors

    def test_compatibility_gguf(self):
        result = extract_pat(
            model_id="TheBloke/Model-GGUF",
            library_name="gguf",
        )
        anchors = [a[0] for a in result.anchors]
        assert "GGUF-available" in anchors
        assert result.compatibility.depth >= 2

    def test_lineage_base_model(self):
        result = extract_pat(
            model_id="meta-llama/Llama-3.1-8B",
            tags=[],
        )
        assert result.lineage.sign == 0  # base model
        anchors = [a[0] for a in result.anchors]
        assert "base-model" in anchors

    def test_lineage_derivative(self):
        result = extract_pat(
            model_id="user/Llama-3.1-8B-GGUF",
            tags=["base_model:meta-llama/Llama-3.1-8B"],
        )
        assert result.base_model == "meta-llama/Llama-3.1-8B"
        assert result.lineage.sign == 1
        assert result.lineage.depth == 3

    def test_family_detection(self):
        result = extract_pat(model_id="meta-llama/Llama-3.1-8B")
        anchors = [a[0] for a in result.anchors]
        assert "Llama-family" in anchors

    def test_domain_medical(self):
        result = extract_pat(
            model_id="medmodel/BioMed-7B",
            tags=["medical", "clinical"],
        )
        anchors = [a[0] for a in result.anchors]
        assert "medical-domain" in anchors
        assert result.domain.depth >= 1


class TestVibes:
    def test_from_card_text(self):
        card = """# Model Card

This is a state-of-the-art language model for code generation. It excels at Python.
"""
        vibe = extract_vibe_summary("test/Model", card_text=card)
        assert "code generation" in vibe.lower()

    def test_fallback_from_signals(self):
        vibe = extract_vibe_summary(
            "meta-llama/Llama-3.1-8B-Instruct",
            pipeline_tag="text-generation",
            author="meta-llama",
        )
        assert "Llama" in vibe
        assert "instruction-tuned" in vibe.lower()

    def test_empty_card(self):
        vibe = extract_vibe_summary("test/Model", card_text="")
        assert len(vibe) > 0  # should fall back to signal synthesis
