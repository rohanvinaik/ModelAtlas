"""Tier 1: Deterministic extraction from HF API fields and config metadata.

Extracts bank positions and metadata from structured fields that have
unambiguous mappings: parameter count, architecture type, dates, downloads,
likes, license, etc.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import NamedTuple


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
    source: str = "huggingface"


@dataclass
class BankPosition:
    """A position in one semantic bank."""

    sign: int = 0
    depth: int = 0
    nodes: list[str] = field(default_factory=list)


class AnchorTag(NamedTuple):
    """An anchor with its bank and confidence score.

    NamedTuple is backward-compatible with a[0]/a[1] indexing,
    so existing code treating anchors as (label, bank) tuples
    continues to work.
    """

    label: str
    bank: str
    confidence: float | None = None


@dataclass
class DeterministicResult:
    """Output of deterministic extraction for one model."""

    architecture: BankPosition = field(default_factory=BankPosition)
    efficiency: BankPosition = field(default_factory=BankPosition)
    quality: BankPosition = field(default_factory=BankPosition)
    metadata: dict[str, tuple[str, str]] = field(default_factory=dict)
    anchors: list[AnchorTag] = field(default_factory=list)


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

# model_type string -> (sign, depth, nodes) — fallback when architectures[0] not in _ARCH_MAP
_MODEL_TYPE_MAP: dict[str, tuple[int, int, list[str]]] = {
    "llama": (0, 0, ["decoder-only"]),
    "mistral": (0, 0, ["decoder-only"]),
    "gpt2": (0, 0, ["decoder-only"]),
    "gpt_neo": (0, 0, ["decoder-only"]),
    "gpt_neox": (0, 0, ["decoder-only"]),
    "qwen2": (0, 0, ["decoder-only"]),
    "phi": (0, 0, ["decoder-only"]),
    "phi3": (0, 0, ["decoder-only"]),
    "gemma": (0, 0, ["decoder-only"]),
    "gemma2": (0, 0, ["decoder-only"]),
    "falcon": (0, 0, ["decoder-only"]),
    "opt": (0, 0, ["decoder-only"]),
    "bloom": (0, 0, ["decoder-only"]),
    "t5": (-1, 1, ["encoder-decoder"]),
    "bart": (-1, 1, ["encoder-decoder"]),
    "bert": (-1, 1, ["encoder-only"]),
    "roberta": (-1, 1, ["encoder-only"]),
    "mamba": (1, 2, ["mamba", "ssm"]),
    "rwkv": (1, 2, ["rwkv"]),
    "mixtral": (1, 1, ["mixture-of-experts"]),
    "clip": (1, 1, ["vision-transformer"]),
    "vit": (1, 1, ["vision-transformer"]),
}


@dataclass
class ConfigSignals:
    """All signals extracted from config.json."""

    context_length: int | None = None
    vocab_size: int | None = None
    hidden_size: int | None = None
    num_layers: int | None = None
    num_heads: int | None = None
    num_kv_heads: int | None = None
    intermediate_size: int | None = None
    model_type: str | None = None
    uses_gqa: bool = False
    rope_scaling_type: str | None = None
    quantization_config: str | None = None
    torch_dtype: str | None = None
    structural_fingerprint: str | None = None


# Parameter count -> (sign, depth, anchor_label)
# Regex to extract parameter count like "7B", "1.5b", "70B" from model names.
# Linear-time: \d+ matches digits, optional \.\d+ matches decimal, [bB] is literal.
# No nested quantifiers over overlapping character classes — no backtracking risk.
_PARAM_COUNT_RE = re.compile(r"(\d+\.\d+|\d+)[bB]")  # NOSONAR: S5852 false positive

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
        match = _PARAM_COUNT_RE.search(text)
        if match:
            val = float(match.group(1))
            if 0.1 <= val <= 1000:
                return val
    return None


_PIPELINE_TAG_ARCH_MAP: dict[str, tuple[int, int, list[str]]] = {
    "object-detection": (1, 1, ["vision-transformer"]),
    "image-classification": (1, 1, ["vision-transformer"]),
    "text-to-image": (1, 2, ["diffusion"]),
    "fill-mask": (-1, 1, ["encoder-only"]),
    "token-classification": (-1, 1, ["encoder-only"]),
    "sentence-similarity": (-1, 1, ["encoder-only"]),
    "translation": (-1, 1, ["encoder-decoder"]),
    "summarization": (-1, 1, ["encoder-decoder"]),
    "automatic-speech-recognition": (-1, 1, ["encoder-decoder"]),
}


def _extract_architecture(
    config: dict | None,
    pipeline_tag: str = "",
    library_name: str = "",
) -> BankPosition:
    """Map model architecture class to ARCHITECTURE bank position.

    Tries architectures[0] in _ARCH_MAP first, then falls back to
    model_type in _MODEL_TYPE_MAP, then pipeline_tag heuristic.
    """
    arch_type = None
    if config and "architectures" in config:
        arch_list = config["architectures"]
        if arch_list:
            arch_type = arch_list[0]

    if arch_type and arch_type in _ARCH_MAP:
        sign, depth, nodes = _ARCH_MAP[arch_type]
        return BankPosition(sign=sign, depth=depth, nodes=list(nodes))

    # Fallback: model_type string
    if config:
        model_type = config.get("model_type")
        if model_type and model_type in _MODEL_TYPE_MAP:
            sign, depth, nodes = _MODEL_TYPE_MAP[model_type]
            return BankPosition(sign=sign, depth=depth, nodes=list(nodes))

    # Fallback: pipeline_tag heuristic
    if pipeline_tag and pipeline_tag in _PIPELINE_TAG_ARCH_MAP:
        sign, depth, nodes = _PIPELINE_TAG_ARCH_MAP[pipeline_tag]
        return BankPosition(sign=sign, depth=depth, nodes=list(nodes))

    return BankPosition(sign=0, depth=0, nodes=["unknown"])


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


def _compute_structural_fingerprint(cfg: ConfigSignals) -> str | None:
    """Hash key architectural dimensions into a stable fingerprint.

    Models with the same fingerprint likely share architecture (e.g. same
    base model with different fine-tunes). Returns None when insufficient
    dimensions are available.
    """
    parts = [cfg.hidden_size, cfg.num_layers, cfg.num_heads, cfg.vocab_size]
    if any(p is None for p in parts):
        return None
    raw = f"{cfg.hidden_size}:{cfg.num_layers}:{cfg.num_heads}:{cfg.vocab_size}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _extract_from_config(config: dict | None) -> ConfigSignals:
    """Extract all structured signals from config.json."""
    if not config:
        return ConfigSignals()

    # Context length: try multiple keys in priority order
    context_length = None
    for key in (
        "max_position_embeddings",
        "max_seq_len",
        "n_positions",
        "max_sequence_length",
        "sliding_window",
    ):
        val = config.get(key)
        if isinstance(val, int) and val > 0:
            context_length = val
            break

    vocab_size = config.get("vocab_size")
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        vocab_size = None

    def _int_or_none(key: str) -> int | None:
        v = config.get(key)
        return v if isinstance(v, int) and v > 0 else None

    hidden_size = _int_or_none("hidden_size")
    num_layers = _int_or_none("num_hidden_layers")
    num_heads = _int_or_none("num_attention_heads")
    num_kv_heads = _int_or_none("num_key_value_heads")
    intermediate_size = _int_or_none("intermediate_size")

    model_type = config.get("model_type")
    if not isinstance(model_type, str):
        model_type = None

    # GQA detection: fewer KV heads than attention heads
    uses_gqa = (
        num_heads is not None and num_kv_heads is not None and num_kv_heads < num_heads
    )

    # RoPE scaling
    rope_scaling_type = None
    rope_cfg = config.get("rope_scaling")
    if isinstance(rope_cfg, dict):
        rt = rope_cfg.get("type") or rope_cfg.get("rope_type")
        if isinstance(rt, str):
            rope_scaling_type = rt.lower()

    # Quantization config
    quantization_config = None
    qcfg = config.get("quantization_config")
    if isinstance(qcfg, dict):
        quantization_config = qcfg.get("quant_method")
        if not isinstance(quantization_config, str):
            quantization_config = None

    torch_dtype = config.get("torch_dtype")
    if not isinstance(torch_dtype, str):
        torch_dtype = None

    cfg = ConfigSignals(
        context_length=context_length,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        model_type=model_type,
        uses_gqa=uses_gqa,
        rope_scaling_type=rope_scaling_type,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
    )
    cfg.structural_fingerprint = _compute_structural_fingerprint(cfg)
    return cfg


def _config_anchors(cfg: ConfigSignals) -> list[AnchorTag]:
    """Generate anchors from config signals."""
    anchors: list[AnchorTag] = []
    if cfg.uses_gqa:
        anchors.append(AnchorTag("grouped-query-attention", "ARCHITECTURE"))
    if cfg.rope_scaling_type:
        label = f"rope-{cfg.rope_scaling_type}"
        anchors.append(AnchorTag(label, "ARCHITECTURE"))
    if cfg.quantization_config:
        label = f"{cfg.quantization_config}-quantized"
        anchors.append(AnchorTag(label, "EFFICIENCY"))
    return anchors


# License string -> anchor label (COMPATIBILITY/license)
_LICENSE_ANCHOR_MAP: dict[str, str] = {
    "apache-2.0": "commercial-use-allowed",
    "mit": "commercial-use-allowed",
    "bsd-2-clause": "commercial-use-allowed",
    "bsd-3-clause": "commercial-use-allowed",
    "openrail": "commercial-use-allowed",
    "openrail++": "commercial-use-allowed",
    "cc-by-4.0": "commercial-use-allowed",
    "cc-by-nc-4.0": "research-only",
    "cc-by-nc-sa-4.0": "research-only",
    "cc-by-nc-nd-4.0": "research-only",
    "cc-by-nc-3.0": "research-only",
    "cc-by-nc-sa-3.0": "research-only",
    "llama3.1": "llama-license",
    "llama3": "llama-license",
    "llama3.2": "llama-license",
    "llama2": "llama-license",
}


def _license_anchors(license_str: str) -> list[AnchorTag]:
    """Map license string to COMPATIBILITY anchors."""
    if not license_str:
        return []
    key = license_str.lower().strip()
    label = _LICENSE_ANCHOR_MAP.get(key)
    if label:
        return [AnchorTag(label, "COMPATIBILITY")]
    return []


@lru_cache(maxsize=256)
def _context_length_anchors(context_length: int | None) -> list[str]:
    """Generate context-length tier anchors."""
    if context_length is None:
        return []
    anchors = []
    if context_length >= 32_768:
        anchors.append("long-context-32k")
    if context_length >= 131_072:
        anchors.append("long-context-128k")
    if context_length >= 1_000_000:
        anchors.append("long-context-1m")
    return anchors


def _collect_metadata(
    inp: ModelInput,
    param_b: float | None,
    cfg: ConfigSignals | None = None,
) -> dict[str, tuple[str, str]]:
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
    if cfg:
        if cfg.context_length is not None:
            meta["context_length"] = (str(cfg.context_length), "int")
        if cfg.vocab_size is not None:
            meta["vocab_size"] = (str(cfg.vocab_size), "int")
        if cfg.hidden_size is not None:
            meta["hidden_size"] = (str(cfg.hidden_size), "int")
        if cfg.num_layers is not None:
            meta["num_layers"] = (str(cfg.num_layers), "int")
        if cfg.num_heads is not None:
            meta["num_heads"] = (str(cfg.num_heads), "int")
        if cfg.num_kv_heads is not None:
            meta["num_kv_heads"] = (str(cfg.num_kv_heads), "int")
        if cfg.intermediate_size is not None:
            meta["intermediate_size"] = (str(cfg.intermediate_size), "int")
        if cfg.model_type:
            meta["model_type"] = (cfg.model_type, "str")
        if cfg.torch_dtype:
            meta["torch_dtype"] = (cfg.torch_dtype, "str")
        if cfg.structural_fingerprint:
            meta["structural_fingerprint"] = (cfg.structural_fingerprint, "str")
    return meta


def extract(inp: ModelInput) -> DeterministicResult:
    """Extract deterministic signals from HF API model metadata."""
    # Architecture
    arch = _extract_architecture(inp.config, inp.pipeline_tag, inp.library_name)

    # Config-based signals
    cfg = _extract_from_config(inp.config)
    ctx_anchors = _context_length_anchors(cfg.context_length)
    cfg_anchors = _config_anchors(cfg)

    # Efficiency
    param_b = _estimate_params_billions(inp.model_id, inp.tags, inp.safetensors_info)
    efficiency, eff_anchors = _extract_efficiency(param_b)

    # Quality
    quality, qual_anchors = _extract_quality(inp.likes, inp.downloads, inp.created_at)

    # License anchors
    lic_anchors = _license_anchors(inp.license_str)

    # Combine anchors
    all_anchors: list[AnchorTag] = [AnchorTag(n, "ARCHITECTURE") for n in arch.nodes]
    all_anchors.extend(AnchorTag(a, "EFFICIENCY") for a in eff_anchors)
    all_anchors.extend(AnchorTag(a, "EFFICIENCY") for a in ctx_anchors)
    all_anchors.extend(AnchorTag(a, "QUALITY") for a in qual_anchors)
    all_anchors.extend(cfg_anchors)
    all_anchors.extend(lic_anchors)

    return DeterministicResult(
        architecture=arch,
        efficiency=efficiency,
        quality=quality,
        metadata=_collect_metadata(inp, param_b, cfg),
        anchors=all_anchors,
    )
