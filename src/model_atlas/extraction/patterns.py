"""Tier 2: Pattern matching on tags, model names, and file lists.

Extracts CAPABILITY, COMPATIBILITY, LINEAGE, DOMAIN, and TRAINING bank
positions plus anchors from signals that require heuristic matching
rather than direct field mapping.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

from .deterministic import AnchorTag, BankPosition


@dataclass
class PatternResult:
    """Output of pattern-based extraction for one model."""

    capability: BankPosition = field(default_factory=BankPosition)
    compatibility: BankPosition = field(default_factory=BankPosition)
    lineage: BankPosition = field(default_factory=BankPosition)
    domain: BankPosition = field(default_factory=BankPosition)
    training: BankPosition = field(default_factory=BankPosition)
    anchors: list[AnchorTag] = field(default_factory=list)
    base_models: list[tuple[str, str]] = field(
        default_factory=list
    )  # (model_id, relation)
    metadata: dict[str, tuple[str, str]] = field(default_factory=dict)

    @property
    def base_model(self) -> str | None:
        """Backward-compatible accessor: first base model ID or None."""
        return self.base_models[0][0] if self.base_models else None


# Tag/name patterns -> (anchor_label, bank)
_CAPABILITY_PATTERNS: list[tuple[str, str]] = [
    (r"\binstruct", "instruction-following"),
    (r"\bchat\b", "chat"),
    (r"\btool[\s_.-]?(?:call|use)", "tool-calling"),
    (r"\bfunction[\s_.-]?call", "function-calling"),
    (r"\bcode\b", "code-generation"),
    (r"\breason", "reasoning"),
    (r"\bmath\b", "math"),
    (r"\bner\b", "NER"),
    (r"\bembedd?ing", "embedding"),
    (r"\bclassif", "classification"),
    (r"\btranslat", "translation"),
    (r"\bsummar", "summarization"),
    (r"\bqa\b|\bquestion[\s_.-]?answer", "question-answering"),
    (r"\bvision\b|\bimage[\s_.-]?understand|\bvlm\b", "image-understanding"),
    (r"\bmultimodal\b", "multimodal"),
    (r"\blong[\s_.-]?context\b|\b\d+k\b", "long-context"),
    (r"\bstruct\w*?[\s_.-]?output\b|\bjson\b", "structured-output"),
    (r"json[\s_-]?mode|structured[\s_-]?output|json[\s_-]?schema", "schema-following"),
    (
        r"\boutlines\b|\bguidance\b|\blmql\b|grammar[\s_-]?constrained",
        "constrained-generation",
    ),
    (r"theorem[\s_-]?prov|formal[\s_-]?math|lean4?|coq|isabelle", "proof-level-math"),
    (r"olympiad|competition[\s_-]?math|imo|aime|putnam", "olympiad-math"),
]

# Quantization level patterns (from model ID)
_QUANT_PATTERN = re.compile(
    r"(?:^|[-_.])"
    r"(Q[2-8]_[KS0-9]+(?:_[SML])?|Q[2-8]_0|F16|F32|"
    r"GPTQ|AWQ|EXL2|GGUF)"
    r"(?:[-_.]|$)",
    re.IGNORECASE,
)

# Language tag detection (2-letter ISO codes in tags)
_LANG_TAG_PATTERN = re.compile(r"^[a-z]{2}$")

_COMPATIBILITY_PATTERNS: list[tuple[str, str]] = [
    (r"\bgguf\b", "GGUF-available"),
    (r"\bgptq\b", "GPTQ-available"),
    (r"\bawq\b", "AWQ-available"),
    (r"\bexl2\b", "EXL2-available"),
    (r"\bonnx\b", "ONNX-available"),
    (r"\bsafetensors\b", "safetensors"),
    (r"\bmlx\b", "MLX-compatible"),
    (r"\bllama[._-]?cpp\b", "llama-cpp-compatible"),
    (r"\bvllm\b", "vLLM-compatible"),
    (r"\btensorrt\b", "TensorRT-compatible"),
    (r"\btransformers\b", "transformers-compatible"),
    (r"\bdiffusers\b", "diffusers-compatible"),
    (r"\bopenvino\b", "OpenVINO"),
    (r"\bcoreml\b|\bcore[\s_-]ml\b", "CoreML"),
    (r"\btflite\b|\btensorflow[\s_-]lite\b", "TFLite"),
    (r"\bcpu[\s_-]?(?:inference|optimized|only)\b", "CPU-inference"),
]

_DOMAIN_PATTERNS: list[tuple[str, str, int]] = [
    # (pattern, anchor, domain_depth)
    (r"\bcode\b|\bcoder\b", "code-domain", 1),
    (r"\bmedic|\bclinical\b|\bbiomed", "medical-domain", 1),
    (r"\blegal\b|\blaw\b", "legal-domain", 1),
    (r"\bfinanc", "finance-domain", 1),
    (r"\bscien|\bchemist|\bphysic|\bbio\b", "science-domain", 1),
    (r"\bmath\b|\barithm", "math-domain", 1),
    (r"\bmultilingual\b|\btranslat", "multilingual", 1),
    (r"\bcreat(?:iv|ion)|\bstory|\bpoet|\broleplay", "creative-domain", 1),
    (r"\bpython\b|\bpy\b", "Python-code", 2),
    (r"\brust\b", "Rust-code", 2),
    (r"\bc\+\+|\bcpp\b", "C++-code", 2),
    (r"\b(?:javascript|js)\b(?!.*typescript)", "JavaScript-code", 2),
    (r"\btypescript\b", "TypeScript-code", 2),
    (r"\b(?:golang|go-lang)\b", "Go-code", 2),
    (r"\bjava\b(?!script)", "Java-code", 2),
    (r"\bradiol|\bpathol", "medical-domain", 2),
    (r"\bsystems[\s_-]?program", "systems-programming", 2),
    (r"\bweb[\s_-]?dev", "web-development", 2),
    (r"\bproof[\s_-]?assist|\blean4?\b|\bcoq\b|\bisabelle\b", "proof-assistant", 2),
    (r"\bformal[\s_-]?verif", "formal-verification", 2),
]

# Family detection: author/name prefix -> family anchor
_FAMILY_MAP: dict[str, str] = {
    "llama": "Llama-family",
    "mistral": "Mistral-family",
    "mixtral": "Mistral-family",
    "qwen": "Qwen-family",
    "phi": "Phi-family",
    "gemma": "Gemma-family",
    "gpt": "GPT-family",
    "falcon": "Falcon-family",
    "stablelm": "StableLM-family",
    "deepseek": "DeepSeek-family",
    "yi": "Yi-family",
    "command": "Command-family",
}


@lru_cache(maxsize=256)
def _detect_capabilities(searchable: str) -> list[str]:
    """Match capability patterns against searchable text."""
    found: list[str] = []
    for pattern, anchor in _CAPABILITY_PATTERNS:
        if re.search(pattern, searchable, re.IGNORECASE):
            found.append(anchor)
    return found


@lru_cache(maxsize=256)
def _detect_compatibility(searchable: str, library_name: str) -> tuple[list[str], int]:
    """Detect format/framework compatibility. Returns anchors and max depth."""
    found: list[str] = []
    max_depth = 0

    for pattern, anchor in _COMPATIBILITY_PATTERNS:
        if re.search(pattern, searchable, re.IGNORECASE):
            found.append(anchor)

    # Library name is a strong compatibility signal
    lib_lower = library_name.lower()
    if "gguf" in lib_lower:
        found.append("GGUF-available")
    if "mlx" in lib_lower:
        found.append("MLX-compatible")
    if "transformers" in lib_lower:
        found.append("transformers-compatible")
    if "diffusers" in lib_lower:
        found.append("diffusers-compatible")

    found = list(set(found))  # deduplicate

    # Depth: how specific is the compatibility target
    format_anchors = {
        "GGUF-available",
        "GPTQ-available",
        "AWQ-available",
        "EXL2-available",
    }
    hw_anchors = {"MLX-compatible", "Apple-Silicon-native", "TensorRT-compatible"}
    if found:
        if any(a in hw_anchors for a in found):
            max_depth = 3
        elif any(a in format_anchors for a in found):
            max_depth = 2
        else:
            max_depth = 1

    return found, max_depth


@lru_cache(maxsize=256)
def _detect_domain(searchable: str) -> tuple[list[str], int]:
    """Detect domain specialization. Returns anchors and max depth."""
    found: list[str] = []
    max_depth = 0
    for pattern, anchor, depth in _DOMAIN_PATTERNS:
        if re.search(pattern, searchable, re.IGNORECASE):
            if anchor not in found:
                found.append(anchor)
            max_depth = max(max_depth, depth)
    return found, max_depth


# Training method patterns -> (anchor, sign, depth)
_TRAINING_PATTERNS: list[tuple[str, str, int, int]] = [
    # (pattern, anchor, sign, depth)
    (r"\brlhf\b", "rlhf-trained", 1, 2),
    (r"\bdpo\b", "dpo-trained", 1, 1),
    (r"\bppo\b", "ppo-trained", 1, 2),
    (r"\borpo\b", "orpo-trained", 1, 1),
    (r"\bkto\b", "kto-trained", 1, 1),
    (r"\bsft\b|\bsupervised[\s_-]?fine[\s_-]?tun", "sft-trained", 0, 0),
    (r"\blora\b(?!.*qlora)", "lora-adapted", -1, 1),
    (r"\bqlora\b", "qlora-adapted", -1, 1),
    (r"\badapter\b", "adapter-tuned", -1, 1),
    (r"\bdistill", "distilled", -1, 2),
    (r"\bquantization[\s_-]?aware", "quantization-aware-trained", -1, 2),
    (r"\bmulti[\s_-]?stage[\s_-]?align", "multi-stage-alignment", 1, 3),
    (r"\bcurriculum", "curriculum-trained", 1, 2),
    (r"\bcontinual[\s_-]?pre[\s_-]?train", "continual-pretrained", 0, 1),
    (r"\balign(?:ed|ment)\b", "rlhf-trained", 1, 1),
]

_TRAINING_DATA_PATTERNS: list[tuple[str, str]] = [
    (r"\bsynthetic[\s_-]?data\b", "trained-on-synthetic-data"),
    (r"\bhuman[\s_-]?feedback\b", "trained-on-human-feedback"),
]

_DATASET_KEYWORDS: set[str] = {
    "alpaca",
    "sharegpt",
    "orca",
    "dolly",
    "oasst",
    "openassistant",
    "ultrachat",
    "wildchat",
    "capybara",
    "slimorca",
    "hermes",
    "platypus",
    "wizard",
    "evol-instruct",
    "metamath",
    "gsm8k",
    "code-feedback",
    "magicoder",
    "openhermes",
    "deita",
}


@lru_cache(maxsize=256)
def _detect_training(searchable: str) -> tuple[list[str], BankPosition, list[str]]:
    """Detect training methodology signals.

    Returns (method_anchors, bank_position, data_anchors).
    """
    method_anchors: list[str] = []
    max_depth = 0
    best_sign = 0

    for pattern, anchor, sign, depth in _TRAINING_PATTERNS:
        if re.search(pattern, searchable, re.IGNORECASE):
            if anchor not in method_anchors:
                method_anchors.append(anchor)
            if depth > max_depth:
                max_depth = depth
                best_sign = sign

    data_anchors: list[str] = []
    for pattern, anchor in _TRAINING_DATA_PATTERNS:
        if re.search(pattern, searchable, re.IGNORECASE):
            data_anchors.append(anchor)

    return method_anchors, BankPosition(sign=best_sign, depth=max_depth), data_anchors


def _detect_training_datasets(searchable: str, tags: list[str]) -> list[str]:
    """Extract training dataset names from tags and text."""
    datasets: list[str] = []

    # Check dataset: tags
    for tag in tags:
        if tag.startswith("dataset:"):
            datasets.append(tag[len("dataset:") :].strip())

    # Keyword matching in searchable text
    searchable_lower = searchable.lower()
    for keyword in _DATASET_KEYWORDS:
        if keyword in searchable_lower:
            datasets.append(keyword)

    return list(set(datasets))


def _compute_card_quality(card_text: str) -> float:
    """Compute a completeness score (0-1) for a model card.

    Checks for presence of key sections: description, usage, training,
    evaluation, limitations, license.
    """
    if not card_text:
        return 0.0

    sections = [
        r"(?:##?\s*(?:description|about|overview|introduction))",
        r"(?:##?\s*(?:usage|how to use|getting started|quick start))",
        r"(?:##?\s*(?:training|fine-?tuning|methodology))",
        r"(?:##?\s*(?:evaluation|benchmark|results|performance))",
        r"(?:##?\s*(?:limitation|bias|risk|ethical))",
        r"(?:##?\s*(?:license|citation|acknowledgment))",
    ]
    found = sum(1 for s in sections if re.search(s, card_text, re.IGNORECASE))
    return round(found / len(sections), 2)


# base_model: tag subtype -> link relation
_TAG_RELATION_MAP: dict[str, str] = {
    "finetune": "fine_tuned_from",
    "adapter": "fine_tuned_from",
    "merge": "merged_from",
    "quantized": "quantized_from",
}


def _infer_relation(model_id: str) -> str:
    """Infer the link relation from model name heuristics."""
    name_lower = model_id.lower()
    if any(q in name_lower for q in ("gguf", "gptq", "awq", "exl2")):
        return "quantized_from"
    if any(q in name_lower for q in ("merge", "franken")):
        return "merged_from"
    return "fine_tuned_from"


def _detect_lineage(
    model_id: str, tags: list[str], author: str
) -> tuple[list[tuple[str, str]], list[str], BankPosition]:
    """Detect model family and lineage position.

    Returns (base_models, anchors, lineage_position) where base_models
    is a list of (model_id, relation) tuples.
    """
    anchors: list[str] = []
    searchable = f"{model_id} {author}".lower()

    # Family detection
    for prefix, family_anchor in _FAMILY_MAP.items():
        if prefix in searchable:
            anchors.append(family_anchor)
            break

    # Base model detection from tags — collect ALL, parse subtypes
    base_models: list[tuple[str, str]] = []
    for tag in tags:
        if not tag.startswith("base_model:"):
            continue
        rest = tag[len("base_model:") :]
        # Parse subtype: base_model:finetune:org/model -> subtype=finetune, id=org/model
        parts = rest.split(":", 1)
        if len(parts) == 2 and parts[0] in _TAG_RELATION_MAP:
            subtype, model_ref = parts
            relation = _TAG_RELATION_MAP[subtype]
            base_models.append((model_ref.strip(), relation))
        else:
            # No subtype or unknown subtype — the whole rest is the model ID
            base_models.append((rest.strip(), _infer_relation(model_id)))

    # Lineage depth heuristics
    name_lower = model_id.lower()
    if base_models:
        # Has base model(s) → derivative
        if any(q in name_lower for q in ("gguf", "gptq", "awq", "exl2")):
            anchors.append("quantized")
            return base_models, anchors, BankPosition(sign=1, depth=3)
        if any(q in name_lower for q in ("merge", "franken")):
            anchors.append("merge")
            return base_models, anchors, BankPosition(sign=1, depth=3)
        if any(q in name_lower for q in ("lora", "dpo", "rlhf", "sft")):
            anchors.append("fine-tune")
            return base_models, anchors, BankPosition(sign=1, depth=2)
        if any(q in name_lower for q in ("instruct", "chat")):
            anchors.append("fine-tune")
            return base_models, anchors, BankPosition(sign=1, depth=1)
        anchors.append("fine-tune")
        return base_models, anchors, BankPosition(sign=1, depth=2)

    # No base model detected — check if it looks like a base model
    if not any(q in name_lower for q in ("instruct", "chat", "gguf", "gptq", "lora")):
        anchors.append("base-model")
        return [], anchors, BankPosition(sign=0, depth=0)

    return [], anchors, BankPosition(sign=1, depth=1)


@lru_cache(maxsize=256)
def _detect_quantization_level(model_id: str) -> str | None:
    """Detect quantization level from model ID."""
    match = _QUANT_PATTERN.search(model_id)
    if match:
        return match.group(1).upper()
    return None


def _detect_chat_template(tags: list[str]) -> bool:
    """Detect chat template availability from tags."""
    for tag in tags:
        tag_lower = tag.lower()
        if any(
            kw in tag_lower
            for kw in ("chat_template", "chat-template", "conversational")
        ):
            return True
    return False


def _detect_language_tags(tags: list[str]) -> list[str]:
    """Extract 2-letter language codes from tags."""
    langs = []
    for tag in tags:
        if _LANG_TAG_PATTERN.match(tag.lower()):
            langs.append(tag.lower())
    return langs


def extract(
    model_id: str,
    author: str = "",
    tags: list[str] | None = None,
    library_name: str = "",
    pipeline_tag: str = "",
) -> PatternResult:
    """Extract pattern-based signals from model metadata."""
    tags = tags or []
    searchable = " ".join([model_id, author, pipeline_tag, *tags]).lower()

    # Capability
    cap_anchors = _detect_capabilities(searchable)
    cap_depth = min(len(cap_anchors), 3)  # more capabilities = richer
    cap_sign = 1 if cap_anchors else 0
    if len(cap_anchors) == 1 and cap_anchors[0] in ("embedding", "classification"):
        cap_sign = -1  # narrow/single-task

    # Compatibility (copy list — cached original must not be mutated)
    compat_anchors, compat_depth = _detect_compatibility(searchable, library_name)
    compat_anchors = list(compat_anchors)

    # Domain
    domain_anchors, domain_depth = _detect_domain(searchable)

    # Lineage
    base_models, lineage_anchors, lineage_pos = _detect_lineage(model_id, tags, author)

    # Training methodology
    training_anchors, training_pos, training_data_anchors = _detect_training(searchable)
    training_datasets = _detect_training_datasets(searchable, tags)

    # Quantization level metadata
    metadata: dict[str, tuple[str, str]] = {}
    quant_level = _detect_quantization_level(model_id)
    if quant_level:
        metadata["quantization_level"] = (quant_level, "str")

    # Training datasets metadata
    if training_datasets:
        metadata["training_datasets"] = (json.dumps(training_datasets), "json")

    # Chat template detection
    has_chat_template = _detect_chat_template(tags)
    if has_chat_template:
        compat_anchors.append("chat-template-available")
        metadata["has_chat_template"] = ("true", "bool")

    # Language tags
    lang_tags = _detect_language_tags(tags)
    if lang_tags:
        metadata["supported_languages"] = (json.dumps(lang_tags), "json")

    # Combine all anchors with per-category confidence
    all_anchors: list[AnchorTag] = []
    all_anchors.extend(AnchorTag(a, "CAPABILITY", 0.85) for a in cap_anchors)
    all_anchors.extend(AnchorTag(a, "COMPATIBILITY", 0.90) for a in compat_anchors)
    all_anchors.extend(AnchorTag(a, "DOMAIN", 0.80) for a in domain_anchors)
    # Lineage: tag-based anchors get 0.95, name-heuristic gets 0.70
    for a in lineage_anchors:
        conf = 0.95 if base_models else 0.70
        all_anchors.append(AnchorTag(a, "LINEAGE", conf))
    # Training: method anchors get 0.85, data anchors get 0.75
    all_anchors.extend(AnchorTag(a, "TRAINING", 0.85) for a in training_anchors)
    all_anchors.extend(AnchorTag(a, "TRAINING", 0.75) for a in training_data_anchors)

    return PatternResult(
        capability=BankPosition(sign=cap_sign, depth=cap_depth),
        compatibility=BankPosition(sign=1 if compat_anchors else 0, depth=compat_depth),
        lineage=lineage_pos,
        domain=BankPosition(sign=1 if domain_anchors else 0, depth=domain_depth),
        training=training_pos,
        anchors=all_anchors,
        base_models=base_models,
        metadata=metadata,
    )
