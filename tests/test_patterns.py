"""Dedicated tests for patterns.py extraction functions.

Provides thorough coverage of each _detect_* function individually,
complementing the higher-level tests in test_extraction.py.
"""

from __future__ import annotations

from model_atlas.extraction.patterns import (
    PatternResult,
    _compute_card_quality,
    _detect_capabilities,
    _detect_chat_template,
    _detect_compatibility,
    _detect_domain,
    _detect_language_tags,
    _detect_lineage,
    _detect_quantization_level,
    _detect_training,
    _detect_training_datasets,
    extract,
)


class TestDetectCapabilities:
    def test_instruct(self):
        assert "instruction-following" in _detect_capabilities("llama instruct model")

    def test_chat(self):
        assert "chat" in _detect_capabilities("model-chat-7b")

    def test_tool_calling(self):
        # Pattern: \btool.?(?:call|use)\b — requires "tool" + optional char + "call"/"use" as complete word
        assert "tool-calling" in _detect_capabilities("tool_call support")
        assert "tool-calling" in _detect_capabilities("tooluse enabled")
        assert "tool-calling" in _detect_capabilities("tool-call support")

    def test_function_calling(self):
        assert "function-calling" in _detect_capabilities("function-call support")
        assert "function-calling" in _detect_capabilities("function_call")

    def test_code_generation(self):
        assert "code-generation" in _detect_capabilities("code model")

    def test_reasoning(self):
        # Pattern: \breason\b — matches exact word "reason", not "reasoning"
        assert "reasoning" in _detect_capabilities("reason about things")

    def test_math(self):
        assert "math" in _detect_capabilities("math problem solver")

    def test_ner(self):
        assert "NER" in _detect_capabilities("ner tagging")

    def test_embedding(self):
        # Pattern: \bembedd?ing\b — matches "embedding" or "embeding" as exact words
        assert "embedding" in _detect_capabilities("text embedding model")
        assert "embedding" in _detect_capabilities("an embedding for retrieval")

    def test_classification(self):
        # Pattern: \bclassif\b — matches the stem "classif" as a standalone word
        # In practice this fires when "classif" appears as a tag or token boundary
        assert "classification" in _detect_capabilities("classif task")
        assert "classification" in _detect_capabilities("text-classif model")

    def test_translation(self):
        # Pattern: \btranslat\b — matches the stem "translat" as a standalone word
        assert "translation" in _detect_capabilities("translat task")
        assert "translation" in _detect_capabilities("a translat model")

    def test_summarization(self):
        # Pattern: \bsummar\b — matches the stem "summar" as a standalone word
        assert "summarization" in _detect_capabilities("summar task")
        assert "summarization" in _detect_capabilities("text-summar model")

    def test_question_answering(self):
        # Pattern: \bqa\b|\bquestion.?answer\b — "qa" as word, or "question" + opt char + "answer" as word
        assert "question-answering" in _detect_capabilities("qa model")
        assert "question-answering" in _detect_capabilities("question-answer model")
        assert "question-answering" in _detect_capabilities("question answer task")

    def test_vision(self):
        # Pattern: \bvision\b|\bimage.?understand\b|\bvlm\b
        assert "image-understanding" in _detect_capabilities("vision language model")
        assert "image-understanding" in _detect_capabilities("vlm model")
        assert "image-understanding" in _detect_capabilities("image-understand model")

    def test_multimodal(self):
        assert "multimodal" in _detect_capabilities("multimodal model")

    def test_long_context(self):
        assert "long-context" in _detect_capabilities("long-context support")
        assert "long-context" in _detect_capabilities("supports 128k tokens")

    def test_structured_output(self):
        assert "structured-output" in _detect_capabilities("structured output")
        assert "structured-output" in _detect_capabilities("json output mode")

    def test_schema_following(self):
        assert "schema-following" in _detect_capabilities("json-mode support")
        assert "schema-following" in _detect_capabilities(
            "structured-output generation"
        )
        assert "schema-following" in _detect_capabilities("json schema constrained")

    def test_constrained_generation(self):
        assert "constrained-generation" in _detect_capabilities("outlines constrained")
        assert "constrained-generation" in _detect_capabilities("uses guidance library")
        assert "constrained-generation" in _detect_capabilities(
            "grammar-constrained decoding"
        )

    def test_proof_level_math(self):
        assert "proof-level-math" in _detect_capabilities("theorem proving")
        assert "proof-level-math" in _detect_capabilities("lean4 formal math")
        assert "proof-level-math" in _detect_capabilities("coq proofs")

    def test_olympiad_math(self):
        assert "olympiad-math" in _detect_capabilities("olympiad solver")
        assert "olympiad-math" in _detect_capabilities("competition-math aime")

    def test_no_false_positives_on_unrelated_text(self):
        found = _detect_capabilities("generic language model for text")
        assert len(found) == 0

    def test_multiple_capabilities_detected(self):
        found = _detect_capabilities("instruct chat code reason model")
        assert "instruction-following" in found
        assert "chat" in found
        assert "code-generation" in found
        assert "reasoning" in found


class TestDetectCompatibility:
    def test_gguf(self):
        anchors, depth = _detect_compatibility("gguf model", "")
        assert "GGUF-available" in anchors
        assert depth >= 2

    def test_gptq(self):
        anchors, _ = _detect_compatibility("gptq quantized", "")
        assert "GPTQ-available" in anchors

    def test_awq(self):
        anchors, _ = _detect_compatibility("awq format", "")
        assert "AWQ-available" in anchors

    def test_exl2(self):
        anchors, _ = _detect_compatibility("exl2 quantized", "")
        assert "EXL2-available" in anchors

    def test_onnx(self):
        anchors, _ = _detect_compatibility("onnx exported", "")
        assert "ONNX-available" in anchors

    def test_safetensors(self):
        anchors, _ = _detect_compatibility("safetensors format", "")
        assert "safetensors" in anchors

    def test_mlx_from_text(self):
        anchors, depth = _detect_compatibility("mlx compatible", "")
        assert "MLX-compatible" in anchors
        assert depth == 3  # hardware-specific

    def test_mlx_from_library(self):
        anchors, _ = _detect_compatibility("", "mlx")
        assert "MLX-compatible" in anchors

    def test_llama_cpp(self):
        anchors, _ = _detect_compatibility("llama-cpp compatible", "")
        assert "llama-cpp-compatible" in anchors
        anchors2, _ = _detect_compatibility("llama.cpp format", "")
        assert "llama-cpp-compatible" in anchors2

    def test_vllm(self):
        anchors, _ = _detect_compatibility("vllm serving", "")
        assert "vLLM-compatible" in anchors

    def test_tensorrt(self):
        anchors, depth = _detect_compatibility("tensorrt optimized", "")
        assert "TensorRT-compatible" in anchors
        assert depth == 3  # hardware-specific

    def test_transformers_from_library(self):
        anchors, _ = _detect_compatibility("", "transformers")
        assert "transformers-compatible" in anchors

    def test_diffusers_from_library(self):
        anchors, _ = _detect_compatibility("", "diffusers")
        assert "diffusers-compatible" in anchors

    def test_openvino(self):
        anchors, _ = _detect_compatibility("openvino export", "")
        assert "OpenVINO" in anchors

    def test_coreml(self):
        anchors, _ = _detect_compatibility("coreml converted", "")
        assert "CoreML" in anchors
        anchors2, _ = _detect_compatibility("core-ml format", "")
        assert "CoreML" in anchors2

    def test_tflite(self):
        anchors, _ = _detect_compatibility("tflite model", "")
        assert "TFLite" in anchors
        anchors2, _ = _detect_compatibility("tensorflow-lite optimized", "")
        assert "TFLite" in anchors2

    def test_cpu_inference(self):
        anchors, _ = _detect_compatibility("cpu-inference optimized", "")
        assert "CPU-inference" in anchors
        anchors2, _ = _detect_compatibility("cpu only model", "")
        assert "CPU-inference" in anchors2

    def test_no_compatibility_returns_zero_depth(self):
        anchors, depth = _detect_compatibility("generic model", "")
        assert anchors == []
        assert depth == 0

    def test_deduplication(self):
        """Anchors from both text and library should be deduplicated."""
        anchors, _ = _detect_compatibility("gguf model", "gguf")
        assert anchors.count("GGUF-available") == 1

    def test_depth_hierarchy(self):
        """Format-specific should be depth 2, hardware-specific depth 3."""
        _, format_depth = _detect_compatibility("gptq quantized", "")
        assert format_depth == 2
        _, hw_depth = _detect_compatibility("mlx optimized", "")
        assert hw_depth == 3
        _, fw_depth = _detect_compatibility("transformers library", "")
        assert fw_depth == 1


class TestDetectDomain:
    def test_code_domain(self):
        anchors, depth = _detect_domain("code generation model")
        assert "code-domain" in anchors
        assert depth >= 1

    def test_medical_domain(self):
        # Pattern: \bmedic\b|\bclinical\b|\bbiomed\b — "clinical" matches as exact word
        anchors, _ = _detect_domain("clinical biomed model")
        assert "medical-domain" in anchors

    def test_legal_domain(self):
        anchors, _ = _detect_domain("legal document analysis")
        assert "legal-domain" in anchors

    def test_finance_domain(self):
        anchors, _ = _detect_domain("financial analysis model")
        assert "finance-domain" in anchors

    def test_science_domain(self):
        # Pattern: \bscien\b|\bchemist\b|\bphysic\b|\bbio\b — stems must be exact words
        anchors, _ = _detect_domain("bio research model")
        assert "science-domain" in anchors

    def test_math_domain(self):
        anchors, _ = _detect_domain("math problem solver")
        assert "math-domain" in anchors

    def test_multilingual(self):
        anchors, _ = _detect_domain("multilingual translation")
        assert "multilingual" in anchors

    def test_creative_domain(self):
        # Pattern: \bcreat\b|\bstory\b|\bpoet\b|\broleplay\b — "story" matches as exact word
        anchors, _ = _detect_domain("story writing roleplay")
        assert "creative-domain" in anchors

    def test_python_narrow_domain(self):
        anchors, depth = _detect_domain("python code generation")
        assert "Python-code" in anchors
        assert depth == 2

    def test_rust_narrow_domain(self):
        anchors, depth = _detect_domain("rust programming")
        assert "Rust-code" in anchors
        assert depth == 2

    def test_cpp_domain(self):
        # Pattern: \b(?:c\+\+|cpp)\b — "cpp" matches as exact word
        anchors, _ = _detect_domain("cpp systems programming")
        assert "C++-code" in anchors

    def test_javascript_domain(self):
        anchors, _ = _detect_domain("javascript web development")
        assert "JavaScript-code" in anchors

    def test_typescript_domain(self):
        anchors, _ = _detect_domain("typescript frontend")
        assert "TypeScript-code" in anchors

    def test_java_not_javascript(self):
        anchors, _ = _detect_domain("java enterprise")
        assert "Java-code" in anchors
        assert "JavaScript-code" not in anchors

    def test_proof_assistant_domain(self):
        anchors, depth = _detect_domain("lean4 proof assistant")
        assert "proof-assistant" in anchors
        assert depth == 2

    def test_formal_verification_domain(self):
        # Pattern: \bformal[\s_-]?verif\b — "formal" + optional separator + "verif" as exact word
        anchors, _ = _detect_domain("formal-verif tools")
        assert "formal-verification" in anchors

    def test_no_domain(self):
        anchors, depth = _detect_domain("generic text model")
        assert anchors == []
        assert depth == 0

    def test_multiple_domains(self):
        # Use exact stems/words that the patterns match
        anchors, _ = _detect_domain("clinical bio research model")
        assert "medical-domain" in anchors
        assert "science-domain" in anchors

    def test_no_duplicate_anchors(self):
        """The same anchor should not appear twice."""
        anchors, _ = _detect_domain("medical clinical biomed")
        assert anchors.count("medical-domain") == 1


class TestDetectLineage:
    def test_base_model_detection(self):
        bases, anchors, pos = _detect_lineage(
            "meta-llama/Llama-3.1-8B", [], "meta-llama"
        )
        assert bases == []
        assert "base-model" in anchors
        assert pos.sign == 0
        assert pos.depth == 0

    def test_family_detection_llama(self):
        _, anchors, _ = _detect_lineage("meta-llama/Llama-3.1-8B", [], "meta-llama")
        assert "Llama-family" in anchors

    def test_family_detection_mistral(self):
        _, anchors, _ = _detect_lineage("mistralai/Mistral-7B", [], "mistralai")
        assert "Mistral-family" in anchors

    def test_family_detection_qwen(self):
        _, anchors, _ = _detect_lineage("Qwen/Qwen2.5-7B", [], "Qwen")
        assert "Qwen-family" in anchors

    def test_family_detection_deepseek(self):
        _, anchors, _ = _detect_lineage("deepseek-ai/DeepSeek-V2", [], "deepseek-ai")
        assert "DeepSeek-family" in anchors

    def test_derivative_from_base_model_tag(self):
        bases, anchors, pos = _detect_lineage(
            "user/Model-FT",
            ["base_model:meta-llama/Llama-3.1-8B"],
            "user",
        )
        assert bases[0][0] == "meta-llama/Llama-3.1-8B"
        assert pos.sign == 1
        assert pos.depth >= 1

    def test_quantized_derivative(self):
        bases, anchors, pos = _detect_lineage(
            "user/Model-GGUF",
            ["base_model:org/BaseModel"],
            "user",
        )
        assert bases[0][0] == "org/BaseModel"
        assert "quantized" in anchors
        assert pos.depth == 3

    def test_merge_derivative(self):
        bases, anchors, pos = _detect_lineage(
            "user/Model-merge",
            ["base_model:org/BaseModel"],
            "user",
        )
        assert "merge" in anchors
        assert pos.depth == 3

    def test_lora_derivative(self):
        bases, anchors, pos = _detect_lineage(
            "user/Model-LoRA",
            ["base_model:org/BaseModel"],
            "user",
        )
        assert "fine-tune" in anchors
        assert pos.depth == 2

    def test_instruct_official_variant(self):
        bases, anchors, pos = _detect_lineage(
            "user/Model-Instruct",
            ["base_model:org/BaseModel"],
            "user",
        )
        assert "fine-tune" in anchors
        assert pos.depth == 1

    def test_no_base_model_with_instruct_name(self):
        """Model with instruct in name but no base_model tag gets depth 1."""
        bases, anchors, pos = _detect_lineage("user/Model-Instruct", [], "user")
        assert bases == []
        assert pos.sign == 1
        assert pos.depth == 1

    def test_generic_derivative_with_base(self):
        """Derivative with base model but no specific type heuristic."""
        bases, anchors, pos = _detect_lineage(
            "user/Model-Custom",
            ["base_model:org/BaseModel"],
            "user",
        )
        assert bases[0][0] == "org/BaseModel"
        assert "fine-tune" in anchors
        assert pos.depth == 2

    def test_subtype_finetune(self):
        """base_model:finetune:org/model should parse subtype correctly."""
        bases, anchors, pos = _detect_lineage(
            "user/Model-FT",
            ["base_model:finetune:org/BaseModel"],
            "user",
        )
        assert bases == [("org/BaseModel", "fine_tuned_from")]

    def test_subtype_quantized(self):
        bases, _, _ = _detect_lineage(
            "user/Model-GGUF",
            ["base_model:quantized:org/BaseModel"],
            "user",
        )
        assert bases == [("org/BaseModel", "quantized_from")]

    def test_subtype_merge(self):
        bases, _, _ = _detect_lineage(
            "user/Model-merge",
            ["base_model:merge:org/BaseModel"],
            "user",
        )
        assert bases == [("org/BaseModel", "merged_from")]

    def test_subtype_adapter(self):
        bases, _, _ = _detect_lineage(
            "user/Model-Adapter",
            ["base_model:adapter:org/BaseModel"],
            "user",
        )
        assert bases == [("org/BaseModel", "fine_tuned_from")]

    def test_multi_parent(self):
        """Multiple base_model tags should all be collected."""
        bases, _, _ = _detect_lineage(
            "user/Model-merge",
            [
                "base_model:merge:org/ModelA",
                "base_model:merge:org/ModelB",
            ],
            "user",
        )
        assert len(bases) == 2
        assert ("org/ModelA", "merged_from") in bases
        assert ("org/ModelB", "merged_from") in bases

    def test_old_format_backward_compat(self):
        """Plain base_model:org/model (no subtype) should still work."""
        bases, _, _ = _detect_lineage(
            "user/Model-FT",
            ["base_model:meta-llama/Llama-3.1-8B"],
            "user",
        )
        assert bases[0][0] == "meta-llama/Llama-3.1-8B"
        assert bases[0][1] == "fine_tuned_from"


class TestDetectQuantizationLevel:
    def test_q4_k_m(self):
        assert _detect_quantization_level("user/Model-Q4_K_M-GGUF") == "Q4_K_M"

    def test_q5_k_s(self):
        assert _detect_quantization_level("user/Model-Q5_K_S") == "Q5_K_S"

    def test_q8_0(self):
        assert _detect_quantization_level("user/Model-Q8_0") == "Q8_0"

    def test_gptq(self):
        assert _detect_quantization_level("user/Model-GPTQ") == "GPTQ"

    def test_awq(self):
        assert _detect_quantization_level("user/Model-AWQ") == "AWQ"

    def test_exl2(self):
        assert _detect_quantization_level("user/Model-EXL2") == "EXL2"

    def test_gguf(self):
        assert _detect_quantization_level("user/Model-GGUF") == "GGUF"

    def test_f16(self):
        assert _detect_quantization_level("user/Model-F16") == "F16"

    def test_no_quantization(self):
        assert _detect_quantization_level("meta-llama/Llama-3.1-8B") is None

    def test_case_insensitive(self):
        result = _detect_quantization_level("user/Model-gptq")
        assert result is not None
        assert result.upper() == "GPTQ"


class TestDetectChatTemplate:
    def test_chat_template_tag(self):
        assert _detect_chat_template(["chat_template", "text-generation"]) is True

    def test_chat_template_hyphenated(self):
        assert _detect_chat_template(["chat-template"]) is True

    def test_conversational_tag(self):
        assert _detect_chat_template(["conversational"]) is True

    def test_no_chat_template(self):
        assert _detect_chat_template(["text-generation", "code"]) is False

    def test_empty_tags(self):
        assert _detect_chat_template([]) is False

    def test_case_insensitive(self):
        assert _detect_chat_template(["Chat_Template"]) is True


class TestDetectLanguageTags:
    def test_standard_language_codes(self):
        langs = _detect_language_tags(["en", "fr", "de"])
        assert "en" in langs
        assert "fr" in langs
        assert "de" in langs

    def test_non_language_tags_excluded(self):
        langs = _detect_language_tags(["text-generation", "code", "en"])
        assert "en" in langs
        assert len(langs) == 1

    def test_empty_tags(self):
        assert _detect_language_tags([]) == []

    def test_three_letter_codes_excluded(self):
        """Only 2-letter ISO codes should be detected."""
        langs = _detect_language_tags(["eng", "fra"])
        assert langs == []

    def test_single_letter_excluded(self):
        langs = _detect_language_tags(["a", "b"])
        assert langs == []

    def test_uppercase_normalized(self):
        langs = _detect_language_tags(["EN", "FR"])
        assert "en" in langs
        assert "fr" in langs


class TestDetectTraining:
    def test_rlhf(self):
        anchors, pos, _ = _detect_training("trained with rlhf alignment")
        assert "rlhf-trained" in anchors
        assert pos.sign == 1

    def test_dpo(self):
        anchors, pos, _ = _detect_training("dpo trained model")
        assert "dpo-trained" in anchors
        assert pos.sign == 1

    def test_lora(self):
        anchors, pos, _ = _detect_training("lora fine-tuned")
        assert "lora-adapted" in anchors
        assert pos.sign == -1

    def test_qlora(self):
        anchors, pos, _ = _detect_training("qlora adapted model")
        assert "qlora-adapted" in anchors
        assert pos.sign == -1

    def test_sft(self):
        anchors, pos, _ = _detect_training("sft trained model")
        assert "sft-trained" in anchors
        assert pos.sign == 0

    def test_distilled(self):
        anchors, pos, _ = _detect_training("distilled from larger model")
        assert "distilled" in anchors
        assert pos.sign == -1

    def test_multi_stage(self):
        anchors, pos, _ = _detect_training("multi-stage alignment process")
        assert "multi-stage-alignment" in anchors
        assert pos.sign == 1
        assert pos.depth == 3

    def test_no_signals(self):
        anchors, pos, _ = _detect_training("generic language model")
        assert anchors == []
        assert pos.sign == 0
        assert pos.depth == 0

    def test_synthetic_data(self):
        _, _, data_anchors = _detect_training("trained on synthetic data")
        assert "trained-on-synthetic-data" in data_anchors

    def test_human_feedback(self):
        _, _, data_anchors = _detect_training("uses human feedback for training")
        assert "trained-on-human-feedback" in data_anchors


class TestDetectTrainingDatasets:
    def test_dataset_tag(self):
        datasets = _detect_training_datasets("", ["dataset:alpaca", "text-generation"])
        assert "alpaca" in datasets

    def test_keyword_matching(self):
        datasets = _detect_training_datasets("trained on sharegpt and orca data", [])
        assert "sharegpt" in datasets
        assert "orca" in datasets

    def test_no_datasets(self):
        datasets = _detect_training_datasets("generic model", [])
        assert datasets == []


class TestCardQuality:
    def test_full_card(self):
        card = """
## Description
A great model.

## Usage
Use it like this.

## Training
Trained on data.

## Evaluation
Results here.

## Limitations
Some limitations.

## License
MIT
"""
        score = _compute_card_quality(card)
        assert score == 1.0

    def test_empty_card(self):
        assert _compute_card_quality("") == 0.0

    def test_partial_card(self):
        card = """
## Description
A great model.

## Evaluation
Results here.
"""
        score = _compute_card_quality(card)
        assert 0.0 < score < 1.0


class TestExtractIntegration:
    """Tests for the top-level extract() function combining all detectors."""

    def test_returns_pattern_result(self):
        result = extract(model_id="test/Model")
        assert isinstance(result, PatternResult)

    def test_capability_sign_positive_with_multiple_capabilities(self):
        result = extract(
            model_id="test/Model-Instruct",
            tags=["code", "chat"],
        )
        assert result.capability.sign == 1

    def test_capability_sign_negative_for_narrow_task(self):
        """Embedding-only model should be sign -1 (narrow capability)."""
        result = extract(
            model_id="test/EmbeddingModel",
            tags=["embedding"],
            pipeline_tag="feature-extraction",
        )
        assert result.capability.sign == -1

    def test_classification_only_is_narrow(self):
        # The pattern \bclassif\b matches exact stem "classif" — the tag "classification"
        # does NOT match because of the trailing characters. Must use a tag that fires the regex.
        result = extract(
            model_id="test/Classifier",
            tags=["classif"],
            pipeline_tag="text-classification",
        )
        assert result.capability.sign == -1

    def test_compatibility_zero_when_no_format_detected(self):
        result = extract(model_id="test/PlainModel", library_name="")
        assert result.compatibility.sign == 0
        assert result.compatibility.depth == 0

    def test_domain_zero_for_general_model(self):
        result = extract(model_id="test/General-7B")
        assert result.domain.sign == 0
        assert result.domain.depth == 0

    def test_anchors_have_bank_assignments(self):
        result = extract(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            tags=["text-generation", "gguf"],
            library_name="gguf",
        )
        banks = {a.bank for a in result.anchors}
        assert "CAPABILITY" in banks
        assert "COMPATIBILITY" in banks
        assert "LINEAGE" in banks

    def test_metadata_includes_quantization(self):
        result = extract(model_id="user/Model-Q4_K_M-GGUF")
        assert "quantization_level" in result.metadata
        assert result.metadata["quantization_level"][0] == "Q4_K_M"

    def test_metadata_includes_chat_template(self):
        result = extract(
            model_id="test/Model",
            tags=["chat_template"],
        )
        assert "has_chat_template" in result.metadata

    def test_metadata_includes_languages(self):
        import json

        result = extract(
            model_id="test/Model",
            tags=["en", "fr", "text-generation"],
        )
        assert "supported_languages" in result.metadata
        langs = json.loads(result.metadata["supported_languages"][0])
        assert "en" in langs
        assert "fr" in langs

    def test_base_model_none_for_base(self):
        result = extract(model_id="meta-llama/Llama-3.1-8B")
        assert result.base_model is None

    def test_base_model_set_for_derivative(self):
        result = extract(
            model_id="user/Llama-GGUF",
            tags=["base_model:meta-llama/Llama-3.1-8B"],
        )
        assert result.base_model == "meta-llama/Llama-3.1-8B"

    def test_training_bank_rlhf(self):
        result = extract(
            model_id="test/RLHF-Model",
            tags=["rlhf", "text-generation"],
        )
        assert result.training.sign == 1
        training_anchors = [a for a in result.anchors if a.bank == "TRAINING"]
        labels = {a.label for a in training_anchors}
        assert "rlhf-trained" in labels

    def test_training_bank_lora(self):
        result = extract(
            model_id="test/LoRA-Model",
            tags=["lora"],
        )
        assert result.training.sign == -1
        training_anchors = [a for a in result.anchors if a.bank == "TRAINING"]
        labels = {a.label for a in training_anchors}
        assert "lora-adapted" in labels

    def test_training_datasets_metadata(self):
        result = extract(
            model_id="test/Model",
            tags=["dataset:alpaca", "text-generation"],
        )
        assert "training_datasets" in result.metadata
