"""Tests for the extraction pipeline."""

from __future__ import annotations

from model_atlas.extraction.deterministic import ModelInput
from model_atlas.extraction.deterministic import extract as extract_det
from model_atlas.extraction.patterns import extract as extract_pat
from model_atlas.extraction.vibes import extract_vibe_summary


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


class TestConfigExtraction:
    def test_context_length_from_max_position_embeddings(self):
        inp = ModelInput(
            model_id="test/Model-7B",
            config={"max_position_embeddings": 32768},
        )
        result = extract_det(inp)
        assert result.metadata["context_length"] == ("32768", "int")
        anchors = [a[0] for a in result.anchors]
        assert "long-context-32k" in anchors

    def test_context_length_from_max_seq_len(self):
        inp = ModelInput(
            model_id="test/Model-7B",
            config={"max_seq_len": 131072},
        )
        result = extract_det(inp)
        assert result.metadata["context_length"] == ("131072", "int")
        anchors = [a[0] for a in result.anchors]
        assert "long-context-32k" in anchors
        assert "long-context-128k" in anchors

    def test_context_length_1m(self):
        inp = ModelInput(
            model_id="test/Model-7B",
            config={"max_position_embeddings": 1048576},
        )
        result = extract_det(inp)
        anchors = [a[0] for a in result.anchors]
        assert "long-context-1m" in anchors

    def test_context_length_short_no_anchors(self):
        inp = ModelInput(
            model_id="test/Model-7B",
            config={"max_position_embeddings": 4096},
        )
        result = extract_det(inp)
        assert result.metadata["context_length"] == ("4096", "int")
        anchors = [a[0] for a in result.anchors]
        assert "long-context-32k" not in anchors

    def test_vocab_size(self):
        inp = ModelInput(
            model_id="test/Model-7B",
            config={"vocab_size": 32000},
        )
        result = extract_det(inp)
        assert result.metadata["vocab_size"] == ("32000", "int")

    def test_no_config(self):
        inp = ModelInput(model_id="test/Model-7B")
        result = extract_det(inp)
        assert "context_length" not in result.metadata
        assert "vocab_size" not in result.metadata


class TestNewPatterns:
    def test_quantization_level_q4km(self):
        result = extract_pat(model_id="user/Model-Q4_K_M-GGUF")
        assert result.metadata.get("quantization_level") == ("Q4_K_M", "str")

    def test_quantization_level_gptq(self):
        result = extract_pat(model_id="user/Model-GPTQ")
        assert result.metadata.get("quantization_level") == ("GPTQ", "str")

    def test_python_code_domain(self):
        result = extract_pat(
            model_id="user/CodeModel",
            tags=["python"],
        )
        anchors = [a[0] for a in result.anchors]
        assert "Python-code" in anchors

    def test_rust_code_domain(self):
        result = extract_pat(
            model_id="user/RustCoder",
            tags=["rust"],
        )
        anchors = [a[0] for a in result.anchors]
        assert "Rust-code" in anchors

    def test_java_not_javascript(self):
        result = extract_pat(
            model_id="user/JavaModel",
            tags=["java"],
        )
        anchors = [a[0] for a in result.anchors]
        assert "Java-code" in anchors
        assert "JavaScript-code" not in anchors

    def test_openvino_detection(self):
        result = extract_pat(
            model_id="user/Model",
            tags=["openvino"],
        )
        anchors = [a[0] for a in result.anchors]
        assert "OpenVINO" in anchors

    def test_coreml_detection(self):
        result = extract_pat(
            model_id="user/Model",
            tags=["coreml"],
        )
        anchors = [a[0] for a in result.anchors]
        assert "CoreML" in anchors

    def test_cpu_inference_detection(self):
        result = extract_pat(
            model_id="user/Model-cpu-inference",
        )
        anchors = [a[0] for a in result.anchors]
        assert "CPU-inference" in anchors

    def test_chat_template_detection(self):
        result = extract_pat(
            model_id="user/Model",
            tags=["chat_template", "text-generation"],
        )
        anchors = [a[0] for a in result.anchors]
        assert "chat-template-available" in anchors
        assert result.metadata.get("has_chat_template") == ("true", "bool")

    def test_constrained_generation(self):
        result = extract_pat(
            model_id="user/Model",
            tags=["outlines", "text-generation"],
        )
        anchors = [a[0] for a in result.anchors]
        assert "constrained-generation" in anchors

    def test_schema_following(self):
        result = extract_pat(
            model_id="user/Model",
            tags=["json-mode", "structured-output"],
        )
        anchors = [a[0] for a in result.anchors]
        assert "schema-following" in anchors

    def test_proof_level_math(self):
        result = extract_pat(
            model_id="user/Lean4-Prover",
            tags=["math", "theorem-proving"],
        )
        anchors = [a[0] for a in result.anchors]
        assert "proof-level-math" in anchors

    def test_olympiad_math(self):
        result = extract_pat(
            model_id="user/OlympiadModel",
            tags=["competition-math", "aime"],
        )
        anchors = [a[0] for a in result.anchors]
        assert "olympiad-math" in anchors

    def test_language_tags(self):
        result = extract_pat(
            model_id="user/Model",
            tags=["en", "fr", "de", "text-generation"],
        )
        import json
        assert "supported_languages" in result.metadata
        langs = json.loads(result.metadata["supported_languages"][0])
        assert "en" in langs
        assert "fr" in langs
        assert "de" in langs


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
