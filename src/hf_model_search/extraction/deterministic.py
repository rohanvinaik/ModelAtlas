"""Tier 1: Deterministic extraction from HF API fields and config metadata.

Extracts bank positions and metadata from structured fields that have
unambiguous mappings: parameter count, architecture type, dates, downloads,
likes, license, etc.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ModelInput:
    """Structured input from the HF API for one model."""

    model_id: str
    author: str = ""
    pipeline_tag: str = ""
    tags: list[str] = field(default_factory=list)
    library_name: str = ""
    likes: int = 0
    downloads: int = 0
    created_at: str | None = None
    license_str: str = ""
    safetensors_info: dict | None = None
    config: dict | None = None


@dataclass
class BankPosition:
    """A position in one semantic bank."""

    sign: int = 0
    depth: int = 0
    nodes: list[str] = field(default_factory=list)


@dataclass
class DeterministicResult:
    """Output of deterministic extraction for one model."""

    architecture: BankPosition = field(default_factory=BankPosition)
    efficiency: BankPosition = field(default_factory=BankPosition)
    quality: BankPosition = field(default_factory=BankPosition)
    metadata: dict[str, tuple[str, str]] = field(default_factory=dict)
    anchors: list[tuple[str, str]] = field(default_factory=list)  # (label, bank)


# Architecture type -> (sign, depth, nodes)
_ARCH_MAP: dict[str, tuple[int, int, list[str]]] = {
    # Negative: simpler/older
    "BertModel": (-1, 1, ["encoder-only"]),
    "RobertaModel": (-1, 1, ["encoder-only"]),
    "AlbertModel": (-1, 1, ["encoder-only"]),
    "DistilBertModel": (-1, 1, ["encoder-only"]),
    "XLNetModel": (-1, 1, ["encoder-only"]),
    "ElectraModel": (-1, 1, ["encoder-only"]),
    "T5ForConditionalGeneration": (-1, 1, ["encoder-decoder"]),
    "BartForConditionalGeneration": (-1, 1, ["encoder-decoder"]),
    "MarianMTModel": (-1, 1, ["encoder-decoder"]),
    # Zero: standard transformer decoder (most common)
    "LlamaForCausalLM": (0, 0, ["decoder-only"]),
    "MistralForCausalLM": (0, 0, ["decoder-only"]),
    "GPT2LMHeadModel": (0, 0, ["decoder-only"]),
    "GPTNeoForCausalLM": (0, 0, ["decoder-only"]),
    "GPTNeoXForCausalLM": (0, 0, ["decoder-only"]),
    "GPTJForCausalLM": (0, 0, ["decoder-only"]),
    "Qwen2ForCausalLM": (0, 0, ["decoder-only"]),
    "PhiForCausalLM": (0, 0, ["decoder-only"]),
    "Phi3ForCausalLM": (0, 0, ["decoder-only"]),
    "GemmaForCausalLM": (0, 0, ["decoder-only"]),
    "Gemma2ForCausalLM": (0, 0, ["decoder-only"]),
    "FalconForCausalLM": (0, 0, ["decoder-only"]),
    "StableLmForCausalLM": (0, 0, ["decoder-only"]),
    "OPTForCausalLM": (0, 0, ["decoder-only"]),
    "BloomForCausalLM": (0, 0, ["decoder-only"]),
    # Positive: novel/specialized
    "MambaForCausalLM": (1, 2, ["mamba", "ssm"]),
    "RwkvForCausalLM": (1, 2, ["rwkv"]),
    "MixtralForCausalLM": (1, 1, ["mixture-of-experts"]),
    "DbrxForCausalLM": (1, 1, ["mixture-of-experts"]),
    "StableDiffusionPipeline": (1, 2, ["diffusion"]),
    "CLIPModel": (1, 1, ["vision-transformer"]),
    "ViTModel": (1, 1, ["vision-transformer"]),
}

# Parameter count -> (sign, depth, anchor_label)
_PARAM_RANGES: list[tuple[float, float, int, int, str]] = [
    (0, 0.5, -1, 3, "sub-1B"),
    (0.5, 1.5, -1, 2, "1B-class"),
    (1.5, 5, -1, 1, "3B-class"),
    (5, 10, 0, 0, "7B-class"),
    (10, 20, 1, 1, "13B-class"),
    (20, 50, 1, 2, "30B-class"),
    (50, 100, 1, 3, "70B-class"),
    (100, float("inf"), 1, 4, "frontier-class"),
]


def _estimate_params_billions(
    model_id: str,
    tags: list[str],
    safetensors_info: dict | None,
) -> float | None:
    """Estimate parameter count in billions from safetensors or model name."""
    # 1. Safetensors metadata (most accurate)
    if safetensors_info:
        params = safetensors_info.get("parameters")
        if isinstance(params, dict):
            return sum(params.values()) / 1e9
        if isinstance(params, (int, float)):
            return params / 1e9
        total = safetensors_info.get("total")
        if isinstance(total, (int, float)):
            return total / 2 / 1e9  # ~2 bytes per param (float16)

    # 2. Parse from model name (7B, 1.5B, 70b, 0.5b)
    for text in [model_id, *tags]:
        match = re.search(r"(\d+\.?\d*)[bB]", text)
        if match:
            val = float(match.group(1))
            if 0.1 <= val <= 1000:
                return val
    return None


def _extract_architecture(config: dict | None) -> BankPosition:
    """Map model architecture class to ARCHITECTURE bank position."""
    arch_type = None
    if config and "architectures" in config:
        arch_list = config["architectures"]
        if arch_list:
            arch_type = arch_list[0]

    if arch_type and arch_type in _ARCH_MAP:
        sign, depth, nodes = _ARCH_MAP[arch_type]
        return BankPosition(sign=sign, depth=depth, nodes=list(nodes))
    return BankPosition(sign=0, depth=0, nodes=["decoder-only"])


def _extract_efficiency(param_b: float | None) -> tuple[BankPosition, list[str]]:
    """Map parameter count to EFFICIENCY bank position and size anchors."""
    anchors: list[str] = []
    if param_b is None:
        return BankPosition(), anchors

    for min_b, max_b, sign, depth, anchor in _PARAM_RANGES:
        if min_b <= param_b < max_b:
            anchors.append(anchor)
            if param_b < 4:
                anchors.append("consumer-GPU-viable")
            if param_b < 1:
                anchors.append("edge-deployable")
            return BankPosition(sign=sign, depth=depth), anchors
    return BankPosition(), anchors


def _extract_quality(
    likes: int, downloads: int, created_at: str | None
) -> tuple[BankPosition, list[str]]:
    """Map popularity signals to QUALITY bank position."""
    anchors: list[str] = []
    pop_score = math.log1p(likes) * 2 + math.log1p(downloads) * 0.5

    # Recency boost
    if created_at:
        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            days = (datetime.now(timezone.utc) - dt).days
            if days < 90:
                pop_score *= 1.5
                anchors.append("trending")
        except (ValueError, TypeError):
            pass

    if downloads > 1_000_000:
        anchors.append("high-downloads")
    if likes > 1000:
        anchors.append("community-favorite")

    if pop_score > 25:
        return BankPosition(sign=1, depth=2), anchors
    if pop_score > 15:
        return BankPosition(sign=1, depth=1), anchors
    if pop_score > 5:
        return BankPosition(sign=0, depth=0), anchors
    if pop_score > 1:
        return BankPosition(sign=-1, depth=1), anchors
    return BankPosition(sign=-1, depth=2), anchors


def _collect_metadata(inp: ModelInput, param_b: float | None) -> dict[str, tuple[str, str]]:
    """Collect overflow metadata fields."""
    meta: dict[str, tuple[str, str]] = {}
    if param_b:
        meta["parameter_count_b"] = (str(round(param_b, 2)), "float")
    if inp.license_str:
        meta["license"] = (inp.license_str, "str")
    if inp.created_at:
        meta["created_at"] = (inp.created_at, "datetime")
    if inp.likes:
        meta["likes"] = (str(inp.likes), "int")
    if inp.downloads:
        meta["downloads"] = (str(inp.downloads), "int")
    if inp.pipeline_tag:
        meta["pipeline_tag"] = (inp.pipeline_tag, "str")
    if inp.library_name:
        meta["library_name"] = (inp.library_name, "str")
    return meta


def extract(inp: ModelInput) -> DeterministicResult:
    """Extract deterministic signals from HF API model metadata."""
    # Architecture
    arch = _extract_architecture(inp.config)
    arch_anchors = [(n, "ARCHITECTURE") for n in arch.nodes]

    # Efficiency
    param_b = _estimate_params_billions(inp.model_id, inp.tags, inp.safetensors_info)
    efficiency, eff_anchors = _extract_efficiency(param_b)

    # Quality
    quality, qual_anchors = _extract_quality(inp.likes, inp.downloads, inp.created_at)

    # Combine anchors
    all_anchors = arch_anchors
    all_anchors.extend((a, "EFFICIENCY") for a in eff_anchors)
    all_anchors.extend((a, "QUALITY") for a in qual_anchors)

    return DeterministicResult(
        architecture=arch,
        efficiency=efficiency,
        quality=quality,
        metadata=_collect_metadata(inp, param_b),
        anchors=all_anchors,
    )
