"""Tier 2: Pattern matching on tags, model names, and file lists.

Extracts CAPABILITY, COMPATIBILITY, LINEAGE, and DOMAIN bank positions
plus anchors from signals that require heuristic matching rather than
direct field mapping.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .deterministic import BankPosition


@dataclass
class PatternResult:
    """Output of pattern-based extraction for one model."""

    capability: BankPosition = field(default_factory=BankPosition)
    compatibility: BankPosition = field(default_factory=BankPosition)
    lineage: BankPosition = field(default_factory=BankPosition)
    domain: BankPosition = field(default_factory=BankPosition)
    anchors: list[tuple[str, str]] = field(default_factory=list)  # (label, bank)
    base_model: str | None = None  # detected base model ID


# Tag/name patterns -> (anchor_label, bank)
_CAPABILITY_PATTERNS: list[tuple[str, str]] = [
    (r"\binstruct\b", "instruction-following"),
    (r"\bchat\b", "chat"),
    (r"\brlhf\b", "RLHF-tuned"),
    (r"\bdpo\b", "DPO-tuned"),
    (r"\btool.?(?:call|use)\b", "tool-calling"),
    (r"\bfunction.?call\b", "function-calling"),
    (r"\bcode\b", "code-generation"),
    (r"\breason\b", "reasoning"),
    (r"\bmath\b", "math"),
    (r"\bner\b", "NER"),
    (r"\bembedd?ing\b", "embedding"),
    (r"\bclassif\b", "classification"),
    (r"\btranslat\b", "translation"),
    (r"\bsummar\b", "summarization"),
    (r"\bqa\b|\bquestion.?answer\b", "question-answering"),
    (r"\bvision\b|\bimage.?understand\b|\bvlm\b", "image-understanding"),
    (r"\bmultimodal\b", "multimodal"),
    (r"\blong.?context\b|\b\d+k\b", "long-context"),
    (r"\bstruct\w*?.?output\b|\bjson\b", "structured-output"),
]

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
]

_DOMAIN_PATTERNS: list[tuple[str, str, int]] = [
    # (pattern, anchor, domain_depth)
    (r"\bcode\b|\bcoder\b", "code-domain", 1),
    (r"\bmedic\b|\bclinical\b|\bbiomed\b", "medical-domain", 1),
    (r"\blegal\b|\blaw\b", "legal-domain", 1),
    (r"\bfinance\b|\bfinancial\b", "finance-domain", 1),
    (r"\bscien\b|\bchemist\b|\bphysic\b|\bbio\b", "science-domain", 1),
    (r"\bmath\b|\barithm\b", "math-domain", 1),
    (r"\bmultilingual\b|\btranslat\b", "multilingual", 1),
    (r"\bcreat\b|\bstory\b|\bpoet\b|\broleplay\b", "creative-domain", 1),
    (r"\bpython\b", "code-domain", 2),  # narrow: specific language
    (r"\bradiol\b|\bpathol\b", "medical-domain", 2),
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


def _detect_capabilities(searchable: str) -> list[str]:
    """Match capability patterns against searchable text."""
    found: list[str] = []
    for pattern, anchor in _CAPABILITY_PATTERNS:
        if re.search(pattern, searchable, re.IGNORECASE):
            found.append(anchor)
    return found


def _detect_compatibility(
    searchable: str, library_name: str
) -> tuple[list[str], int]:
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
    format_anchors = {"GGUF-available", "GPTQ-available", "AWQ-available", "EXL2-available"}
    hw_anchors = {"MLX-compatible", "Apple-Silicon-native", "TensorRT-compatible"}
    if found:
        if any(a in hw_anchors for a in found):
            max_depth = 3
        elif any(a in format_anchors for a in found):
            max_depth = 2
        else:
            max_depth = 1

    return found, max_depth


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


def _detect_lineage(
    model_id: str, tags: list[str], author: str
) -> tuple[str | None, list[str], BankPosition]:
    """Detect model family and lineage position."""
    anchors: list[str] = []
    searchable = f"{model_id} {author}".lower()

    # Family detection
    for prefix, family_anchor in _FAMILY_MAP.items():
        if prefix in searchable:
            anchors.append(family_anchor)
            break

    # Base model detection from tags
    base_model = None
    for tag in tags:
        if tag.startswith("base_model:"):
            base_model = tag.split(":", 1)[1].strip()
            break

    # Lineage depth heuristics
    name_lower = model_id.lower()
    if base_model:
        # Has a base model → derivative
        if any(q in name_lower for q in ["gguf", "gptq", "awq", "exl2"]):
            anchors.append("quantized")
            return base_model, anchors, BankPosition(sign=1, depth=3)
        if any(q in name_lower for q in ["merge", "franken"]):
            anchors.append("merge")
            return base_model, anchors, BankPosition(sign=1, depth=3)
        if any(q in name_lower for q in ["lora", "dpo", "rlhf", "sft"]):
            anchors.append("fine-tune")
            return base_model, anchors, BankPosition(sign=1, depth=2)
        if any(q in name_lower for q in ["instruct", "chat"]):
            anchors.append("fine-tune")
            return base_model, anchors, BankPosition(sign=1, depth=1)
        anchors.append("fine-tune")
        return base_model, anchors, BankPosition(sign=1, depth=2)

    # No base model detected — check if it looks like a base model
    if not any(q in name_lower for q in ["instruct", "chat", "gguf", "gptq", "lora"]):
        anchors.append("base-model")
        return None, anchors, BankPosition(sign=0, depth=0)

    return None, anchors, BankPosition(sign=1, depth=1)


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

    # Compatibility
    compat_anchors, compat_depth = _detect_compatibility(searchable, library_name)

    # Domain
    domain_anchors, domain_depth = _detect_domain(searchable)

    # Lineage
    base_model, lineage_anchors, lineage_pos = _detect_lineage(model_id, tags, author)

    # Combine all anchors
    all_anchors: list[tuple[str, str]] = []
    all_anchors.extend((a, "CAPABILITY") for a in cap_anchors)
    all_anchors.extend((a, "COMPATIBILITY") for a in compat_anchors)
    all_anchors.extend((a, "DOMAIN") for a in domain_anchors)
    all_anchors.extend((a, "LINEAGE") for a in lineage_anchors)

    return PatternResult(
        capability=BankPosition(sign=cap_sign, depth=cap_depth),
        compatibility=BankPosition(sign=1 if compat_anchors else 0, depth=compat_depth),
        lineage=lineage_pos,
        domain=BankPosition(sign=1 if domain_anchors else 0, depth=domain_depth),
        anchors=all_anchors,
        base_model=base_model,
    )
