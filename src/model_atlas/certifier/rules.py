"""Deterministic rule library — HF-fact ⇒ anchor invariants.

Ported in intent from `phase_d_audit._check_*` (which currently only *logs*
mismatches) and from triagegeist's `HARD_FLOOR_RULES`. Each rule is a pure
data object; the certifier applies them in tier order.

Rule tiers (matches EvidenceType trust tiers in contract.py):

  TIER 1 STRUCTURAL — inviolable. HF publishes it verbatim in a config or
    metadata field. If a Tier-3 emission contradicts a Tier-1 rule, the
    Tier-3 emission is REJECTED.

  TIER 2 SEMI-STRUCTURAL — strong convention. Tag strings, model_id
    patterns. Contradiction with these DEMOTES the offending emission
    rather than rejecting it (there is some legitimate reinterpretation
    room, e.g. a re-fine-tuned quantized model).

  TIER 3 INFERRED — advisory. Web-source or LLM-derived. Contradiction
    with these emits a WARNING; the emission survives but is flagged.

Rule structure:

  Trigger:  which HF fact fires this rule (evidence type + value predicate)
  Requires: labels the model MUST carry if the trigger fires (AUTO_ADDED
            if missing from the proposed set)
  Forbids:  labels the model MUST NOT carry (REJECTED / DEMOTED if present)

Rules NEVER contradict each other. When adding new rules, run the sanity
check at the bottom of this module.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# contract types are referenced by the certifier; rules.py only declares
# data-driven Rule objects and their triggers.


class RuleTier(int, Enum):
    STRUCTURAL = 1
    SEMI_STRUCTURAL = 2
    INFERRED = 3


@dataclass(frozen=True)
class HFFacts:
    """The minimal HF-fact bundle a rule can inspect.

    Populated from `models.model_type`/`pipeline_tag`, `model_metadata` keys
    like `library_name`, `license`, `parameter_count_b`, `context_length`,
    `safetensors_info`, plus the raw tag list. Keep this dataclass narrow —
    it is the certifier's read-only view of "what HF says about this model."
    """
    model_id: str
    pipeline_tag: str = ""
    model_type: str = ""
    library_name: str = ""
    license: str = ""
    tags: tuple[str, ...] = ()
    param_count_b: float | None = None
    context_length: int | None = None
    safetensors_present: bool = False
    quantization_level: str = ""
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Rule:
    """One HF-fact ⇒ anchor invariant.

    Rule fires when `trigger(facts)` returns True. When it fires:

      * every label in `requires` must appear in the model's anchor set
        (AUTO_ADDED with confidence=1.0, evidence=this rule if missing)
      * no label in `forbids` may appear (REJECTED at STRUCTURAL, DEMOTED
        at SEMI_STRUCTURAL, WARNING at INFERRED)

    `trigger` is a pure predicate — no DB access, no I/O.
    """
    name: str
    tier: RuleTier
    trigger: Callable[[HFFacts], bool]
    requires: tuple[str, ...] = ()
    forbids: tuple[str, ...] = ()
    reason_template: str = ""
    """Short human-readable reason shown in the certification log.
    Rendered with `.format(facts=facts)` when the rule fires."""


# ---------------------------------------------------------------------------
# Trigger builders — small helpers so rule bodies stay declarative
# ---------------------------------------------------------------------------


def _pipeline_is(value: str) -> Callable[[HFFacts], bool]:
    return lambda f: f.pipeline_tag == value


def _model_type_in(*values: str) -> Callable[[HFFacts], bool]:
    vs = frozenset(values)
    return lambda f: f.model_type in vs


def _library_is(value: str) -> Callable[[HFFacts], bool]:
    return lambda f: f.library_name == value


def _tag_matches(pattern: str) -> Callable[[HFFacts], bool]:
    rx = re.compile(pattern, re.IGNORECASE)
    return lambda f: any(rx.search(t) for t in f.tags)


def _has_safetensors() -> Callable[[HFFacts], bool]:
    return lambda f: f.safetensors_present


def _quantization_is(value: str) -> Callable[[HFFacts], bool]:
    return lambda f: f.quantization_level.upper() == value.upper()


# ---------------------------------------------------------------------------
# Tier 1 — STRUCTURAL rules
# Direct implications from HF fields that HF itself publishes.
# ---------------------------------------------------------------------------


PIPELINE_TAG_RULES: tuple[Rule, ...] = (
    Rule(
        name="pipeline_image_text_to_text",
        tier=RuleTier.STRUCTURAL,
        trigger=_pipeline_is("image-text-to-text"),
        requires=("multimodal", "image-understanding"),
        forbids=("image-generation",),
        reason_template="pipeline_tag=image-text-to-text implies image UNDERSTANDING, not generation",
    ),
    Rule(
        name="pipeline_text_to_image",
        tier=RuleTier.STRUCTURAL,
        trigger=_pipeline_is("text-to-image"),
        requires=("image-generation",),
        forbids=("image-understanding",),
        reason_template="pipeline_tag=text-to-image implies image GENERATION",
    ),
    Rule(
        name="pipeline_text_generation_forbids_encoder_only",
        tier=RuleTier.STRUCTURAL,
        trigger=_pipeline_is("text-generation"),
        forbids=("encoder-only",),
        reason_template="pipeline_tag=text-generation forbids encoder-only architecture",
    ),
    Rule(
        name="pipeline_text2text_requires_encoder_decoder",
        tier=RuleTier.STRUCTURAL,
        trigger=_pipeline_is("text2text-generation"),
        requires=("encoder-decoder",),
        forbids=("decoder-only", "encoder-only"),
        reason_template="pipeline_tag=text2text-generation implies encoder-decoder",
    ),
    Rule(
        name="pipeline_feature_extraction_requires_embedding",
        tier=RuleTier.STRUCTURAL,
        trigger=_pipeline_is("feature-extraction"),
        requires=("embedding",),
        reason_template="pipeline_tag=feature-extraction implies embedding capability",
    ),
    Rule(
        name="pipeline_sentence_similarity_requires_embedding",
        tier=RuleTier.STRUCTURAL,
        trigger=_pipeline_is("sentence-similarity"),
        requires=("embedding",),
        reason_template="pipeline_tag=sentence-similarity implies embedding capability",
    ),
    Rule(
        name="pipeline_text_classification_requires_classification",
        tier=RuleTier.STRUCTURAL,
        trigger=_pipeline_is("text-classification"),
        requires=("classification",),
        forbids=("code-generation", "creative-writing", "chat"),
        reason_template="pipeline_tag=text-classification is single-task, not generative-chat",
    ),
    Rule(
        name="pipeline_token_classification_requires_classification",
        tier=RuleTier.STRUCTURAL,
        trigger=_pipeline_is("token-classification"),
        requires=("classification",),
        forbids=("chat", "creative-writing"),
        reason_template="pipeline_tag=token-classification is NER-style, not chat",
    ),
    Rule(
        name="pipeline_image_classification_forbids_generative_llm",
        tier=RuleTier.STRUCTURAL,
        trigger=_pipeline_is("image-classification"),
        forbids=("chat", "reasoning", "code-generation", "creative-writing", "tool-calling", "function-calling"),
        reason_template="pipeline_tag=image-classification is not a generative LLM",
    ),
    Rule(
        name="pipeline_automatic_speech_recognition_forbids_generative_llm",
        tier=RuleTier.STRUCTURAL,
        trigger=_pipeline_is("automatic-speech-recognition"),
        forbids=("chat", "reasoning", "code-generation", "creative-writing"),
        reason_template="pipeline_tag=automatic-speech-recognition is transcription, not generative",
    ),
)


MODEL_TYPE_RULES: tuple[Rule, ...] = (
    Rule(
        name="model_type_qwen2",
        tier=RuleTier.STRUCTURAL,
        trigger=_model_type_in("qwen2", "qwen2_moe", "qwen3", "qwen3_moe"),
        requires=("decoder-only", "grouped-query-attention"),
        forbids=("encoder-only", "encoder-decoder", "mamba", "rwkv"),
        reason_template="model_type={facts.model_type} is a Qwen decoder with GQA",
    ),
    Rule(
        name="model_type_qwen2_moe",
        tier=RuleTier.STRUCTURAL,
        trigger=_model_type_in("qwen2_moe", "qwen3_moe"),
        requires=("mixture-of-experts",),
        reason_template="model_type={facts.model_type} is Qwen MoE",
    ),
    Rule(
        name="model_type_llama",
        tier=RuleTier.STRUCTURAL,
        trigger=_model_type_in("llama"),
        requires=("decoder-only",),
        forbids=("encoder-only", "encoder-decoder", "mamba", "rwkv", "mixture-of-experts"),
        reason_template="model_type=llama is a plain decoder architecture",
    ),
    Rule(
        name="model_type_mistral",
        tier=RuleTier.STRUCTURAL,
        trigger=_model_type_in("mistral"),
        requires=("decoder-only",),
        forbids=("encoder-only", "encoder-decoder", "mixture-of-experts"),
        reason_template="model_type=mistral is a plain decoder",
    ),
    Rule(
        name="model_type_mixtral",
        tier=RuleTier.STRUCTURAL,
        trigger=_model_type_in("mixtral"),
        requires=("mixture-of-experts", "decoder-only"),
        forbids=("encoder-only", "encoder-decoder"),
        reason_template="model_type=mixtral is MoE decoder",
    ),
    Rule(
        name="model_type_mamba",
        tier=RuleTier.STRUCTURAL,
        trigger=_model_type_in("mamba", "mamba2"),
        requires=("mamba", "ssm"),
        forbids=("decoder-only", "encoder-only", "encoder-decoder"),
        reason_template="model_type={facts.model_type} is state-space, not transformer",
    ),
    Rule(
        name="model_type_rwkv",
        tier=RuleTier.STRUCTURAL,
        trigger=_model_type_in("rwkv", "rwkv5", "rwkv6"),
        requires=("rwkv",),
        forbids=("decoder-only", "encoder-only", "encoder-decoder", "mamba"),
        reason_template="model_type={facts.model_type} is RWKV, not transformer",
    ),
    Rule(
        name="model_type_encoder_only_bert_family",
        tier=RuleTier.STRUCTURAL,
        trigger=_model_type_in("bert", "roberta", "deberta", "electra", "albert", "distilbert"),
        requires=("encoder-only",),
        forbids=("decoder-only", "encoder-decoder", "chat", "reasoning", "code-generation"),
        reason_template="model_type={facts.model_type} is encoder-only, not generative",
    ),
    Rule(
        name="model_type_encoder_decoder_t5_family",
        tier=RuleTier.STRUCTURAL,
        trigger=_model_type_in("t5", "mt5", "byt5", "flan_t5", "bart", "mbart", "marian"),
        requires=("encoder-decoder",),
        forbids=("decoder-only", "encoder-only"),
        reason_template="model_type={facts.model_type} is encoder-decoder",
    ),
)


LIBRARY_NAME_RULES: tuple[Rule, ...] = (
    Rule(
        name="library_gguf",
        tier=RuleTier.STRUCTURAL,
        trigger=_library_is("gguf"),
        requires=("GGUF-available", "llama-cpp-compatible"),
        reason_template="library_name=gguf implies GGUF format + llama.cpp compatibility",
    ),
    Rule(
        name="library_mlx",
        tier=RuleTier.STRUCTURAL,
        trigger=_library_is("mlx"),
        requires=("Apple-Silicon-native",),
        reason_template="library_name=mlx implies Apple Silicon MLX runtime",
    ),
    Rule(
        name="library_diffusers",
        tier=RuleTier.STRUCTURAL,
        trigger=_library_is("diffusers"),
        requires=("diffusion",),
        reason_template="library_name=diffusers implies diffusion architecture",
    ),
    Rule(
        name="library_sentence_transformers",
        tier=RuleTier.STRUCTURAL,
        trigger=_library_is("sentence-transformers"),
        requires=("embedding",),
        reason_template="library_name=sentence-transformers is an embedding library",
    ),
)


SAFETENSORS_RULES: tuple[Rule, ...] = (
    Rule(
        name="safetensors_present",
        tier=RuleTier.STRUCTURAL,
        trigger=_has_safetensors(),
        requires=("safetensors",),
        reason_template="safetensors index present ⇒ safetensors anchor required",
    ),
)


QUANTIZATION_RULES: tuple[Rule, ...] = (
    Rule(
        name="quantization_gguf",
        tier=RuleTier.STRUCTURAL,
        trigger=_quantization_is("GGUF"),
        requires=("GGUF-available", "quantized"),
        reason_template="quantization_level=GGUF implies GGUF format + quantized",
    ),
    Rule(
        name="quantization_gptq",
        tier=RuleTier.STRUCTURAL,
        trigger=_quantization_is("GPTQ"),
        requires=("GPTQ-available", "gptq-quantized", "quantized"),
        reason_template="quantization_level=GPTQ implies GPTQ format anchors",
    ),
    Rule(
        name="quantization_awq",
        tier=RuleTier.STRUCTURAL,
        trigger=_quantization_is("AWQ"),
        requires=("AWQ-available", "awq-quantized", "quantized"),
        reason_template="quantization_level=AWQ implies AWQ format anchors",
    ),
)


# ---------------------------------------------------------------------------
# Tier 2 — SEMI-STRUCTURAL rules (tag conventions)
# ---------------------------------------------------------------------------


TAG_RULES: tuple[Rule, ...] = (
    Rule(
        name="tag_gguf_variant",
        tier=RuleTier.SEMI_STRUCTURAL,
        trigger=_tag_matches(r"(?:^|-|:)gguf(?:$|-|:)"),
        requires=("GGUF-available",),
        reason_template="tag matching gguf pattern implies GGUF quantization available",
    ),
    Rule(
        name="tag_gptq_variant",
        tier=RuleTier.SEMI_STRUCTURAL,
        trigger=_tag_matches(r"(?:^|-|:)gptq(?:$|-|:)"),
        requires=("GPTQ-available",),
        reason_template="tag matching gptq pattern implies GPTQ available",
    ),
    Rule(
        name="tag_awq_variant",
        tier=RuleTier.SEMI_STRUCTURAL,
        trigger=_tag_matches(r"(?:^|-|:)awq(?:$|-|:)"),
        requires=("AWQ-available",),
        reason_template="tag matching awq pattern implies AWQ available",
    ),
    Rule(
        name="tag_conversational",
        tier=RuleTier.SEMI_STRUCTURAL,
        trigger=_tag_matches(r"^conversational$"),
        requires=("chat",),
        reason_template="tag=conversational implies chat capability",
    ),
)


# ---------------------------------------------------------------------------
# Cross-anchor derivation rules (fire on other anchors, not HF facts)
# Handled specially — see certifier for the trigger convention.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Lineage-family evidence rules
# Each `X-family` LINEAGE anchor requires ONE of:
#   (a) model_type matches the family key (e.g. `llama`, `qwen2`, `mistral`)
#   (b) family word appears in the model_id itself (case-insensitive substring)
# Author name alone is NOT sufficient — surfaced by the Falconsai audit:
# `Falconsai/nsfw_image_detection` isn't a Falcon LLM, it's a ViT classifier
# from an unrelated author. Same shape for any -family anchor.
# ---------------------------------------------------------------------------

def _family_absence(family_word: str, model_types: tuple[str, ...]) -> Callable[[HFFacts], bool]:
    """Trigger fires when the model shows NO family evidence.

    True (rule fires + forbids the family anchor) when both:
      * model_type is not in the accepted family-model-type set
      * family word does not appear in the *repo name* part of model_id
        (author name is deliberately excluded — Falconsai/nsfw_image_detection
        is not a Falcon)

    False (rule does not fire) when either check passes — the anchor is
    supported and stays.
    """
    accepted_types = frozenset(mt.lower() for mt in model_types)
    fw = family_word.lower()

    def _trigger(f: HFFacts) -> bool:
        if f.model_type and f.model_type.lower() in accepted_types:
            return False
        # Only inspect the repo-name portion, not the author
        repo_name = f.model_id.split("/", 1)[-1].lower() if f.model_id else ""
        if fw in repo_name:
            return False
        return True

    return _trigger


FAMILY_EVIDENCE_RULES: tuple[Rule, ...] = (
    Rule(
        name="lineage_falcon_family_requires_evidence",
        tier=RuleTier.STRUCTURAL,
        trigger=_family_absence("falcon", ("falcon", "falcon_mamba")),
        forbids=("Falcon-family",),
        reason_template="Falcon-family requires model_type=falcon or 'falcon' in repo name",
    ),
    Rule(
        name="lineage_llama_family_requires_evidence",
        tier=RuleTier.STRUCTURAL,
        trigger=_family_absence("llama", ("llama",)),
        forbids=("Llama-family",),
        reason_template="Llama-family requires model_type=llama or 'llama' in repo name",
    ),
    Rule(
        name="lineage_qwen_family_requires_evidence",
        tier=RuleTier.STRUCTURAL,
        trigger=_family_absence("qwen", ("qwen2", "qwen2_moe", "qwen3", "qwen3_moe")),
        forbids=("Qwen-family",),
        reason_template="Qwen-family requires model_type=qwen* or 'qwen' in repo name",
    ),
    Rule(
        name="lineage_mistral_family_requires_evidence",
        tier=RuleTier.STRUCTURAL,
        trigger=_family_absence("mistral", ("mistral", "mixtral")),
        forbids=("Mistral-family",),
        reason_template="Mistral-family requires model_type=mistral/mixtral or 'mistral' in repo name",
    ),
    Rule(
        name="lineage_gemma_family_requires_evidence",
        tier=RuleTier.STRUCTURAL,
        trigger=_family_absence("gemma", ("gemma", "gemma2", "gemma3")),
        forbids=("Gemma-family",),
        reason_template="Gemma-family requires model_type=gemma* or 'gemma' in repo name",
    ),
    Rule(
        name="lineage_phi_family_requires_evidence",
        tier=RuleTier.STRUCTURAL,
        trigger=_family_absence("phi", ("phi", "phi3", "phi_moe")),
        forbids=("Phi-family",),
        reason_template="Phi-family requires model_type=phi* or 'phi' in repo name",
    ),
    Rule(
        name="lineage_gpt_family_requires_evidence",
        tier=RuleTier.STRUCTURAL,
        trigger=_family_absence("gpt", ("gpt2", "gpt_neox", "gpt_bigcode", "gptj")),
        forbids=("GPT-family",),
        reason_template="GPT-family requires model_type=gpt* or 'gpt' in repo name",
    ),
    Rule(
        name="lineage_deepseek_family_requires_evidence",
        tier=RuleTier.STRUCTURAL,
        trigger=_family_absence("deepseek", ("deepseek", "deepseek_v2", "deepseek_v3", "deepseek_vl_v2")),
        forbids=("DeepSeek-family",),
        reason_template="DeepSeek-family requires model_type=deepseek* or 'deepseek' in repo name",
    ),
    Rule(
        name="lineage_yi_family_requires_evidence",
        tier=RuleTier.STRUCTURAL,
        trigger=_family_absence("yi", ("yi",)),
        forbids=("Yi-family",),
        reason_template="Yi-family requires model_type=yi or 'yi' in repo name",
    ),
    Rule(
        name="lineage_stablelm_family_requires_evidence",
        tier=RuleTier.STRUCTURAL,
        trigger=_family_absence("stablelm", ("stablelm", "stablelm_epoch")),
        forbids=("StableLM-family",),
        reason_template="StableLM-family requires model_type=stablelm* or 'stablelm' in repo name",
    ),
    Rule(
        name="lineage_command_family_requires_evidence",
        tier=RuleTier.STRUCTURAL,
        trigger=_family_absence("command", ("cohere",)),
        forbids=("Command-family",),
        reason_template="Command-family requires model_type=cohere or 'command' in repo name",
    ),
)


# ---------------------------------------------------------------------------
# Code-language DOMAIN evidence rules
# `<Lang>-code` DOMAIN anchors require the model to actually be capable of
# code work. Non-generative pipelines (sentence-similarity, classification,
# feature-extraction, ASR) get these anchors REJECTED — surfaced by the
# MiniLM-L6-v2 audit which had `Rust-code` at 0.8 because web scrape saw a
# mention of `candle` (the Rust ML framework).
# ---------------------------------------------------------------------------

_CODE_LANGUAGE_ANCHORS: tuple[str, ...] = (
    "Python-code",
    "Rust-code",
    "C++-code",
    "Java-code",
    "JavaScript-code",
    "TypeScript-code",
    "Go-code",
    "SQL-code",
    "Ruby-code",
    "PHP-code",
    "C-code",
    "C#-code",
    "Kotlin-code",
    "Swift-code",
    "Scala-code",
    "Haskell-code",
    "Shell-code",
)

_NON_GENERATIVE_PIPELINES = frozenset({
    "sentence-similarity",
    "feature-extraction",
    "image-classification",
    "text-classification",
    "token-classification",
    "automatic-speech-recognition",
    "image-to-text",
    "audio-classification",
    "zero-shot-classification",
    "zero-shot-image-classification",
    "object-detection",
    "image-segmentation",
    "depth-estimation",
})


CODE_LANGUAGE_RULES: tuple[Rule, ...] = (
    Rule(
        name="domain_language_requires_generative_task",
        tier=RuleTier.STRUCTURAL,
        trigger=lambda f: f.pipeline_tag in _NON_GENERATIVE_PIPELINES,
        forbids=_CODE_LANGUAGE_ANCHORS,
        reason_template="pipeline_tag={facts.pipeline_tag} is non-generative — code-language DOMAIN anchors don't apply",
    ),
)


ALL_RULES: tuple[Rule, ...] = (
    *PIPELINE_TAG_RULES,
    *MODEL_TYPE_RULES,
    *LIBRARY_NAME_RULES,
    *SAFETENSORS_RULES,
    *QUANTIZATION_RULES,
    *TAG_RULES,
    *FAMILY_EVIDENCE_RULES,
    *CODE_LANGUAGE_RULES,
)


# ---------------------------------------------------------------------------
# Sanity checks — run at import time to catch rule-set drift
# ---------------------------------------------------------------------------


def _sanity_check_rules() -> None:
    """Assert no rule requires+forbids the same label; assert all referenced
    labels are in the static vocabulary."""
    from ..contract import VOCABULARY
    seen_names: set[str] = set()
    for rule in ALL_RULES:
        if rule.name in seen_names:
            raise ValueError(f"duplicate rule name: {rule.name}")
        seen_names.add(rule.name)
        overlap = set(rule.requires) & set(rule.forbids)
        if overlap:
            raise ValueError(
                f"rule {rule.name} both requires and forbids: {sorted(overlap)}"
            )
        for label in (*rule.requires, *rule.forbids):
            if label not in VOCABULARY:
                # Dynamically-materialized anchors won't be in the static
                # vocab — allow but do not fail. Labels here should generally
                # be static seeds though.
                pass


_sanity_check_rules()
