#!/usr/bin/env python3
"""Phase E post-processor: deterministic regex + structured LLM validation.

Two-layer cleaning pipeline for Phase E web enrichment results:
  Layer 1 (regex): synonym merges, name-based contradictions, pipeline-tag gates, size-class dedup
  Layer 2 (LLM):   series-bleed detection via constrained Ollama validation

Usage:
  # Layer 1 only (fast, no LLM):
  python scripts/phase_e_postprocess.py --input ~/.cache/model-atlas/phase_e_work/results_0.jsonl

  # Both layers:
  python scripts/phase_e_postprocess.py --input ~/.cache/model-atlas/phase_e_work/results_0.jsonl --llm

  # Process all result files:
  python scripts/phase_e_postprocess.py --input-dir ~/.cache/model-atlas/phase_e_work/ --llm

  # Dry run (report only, no output):
  python scripts/phase_e_postprocess.py --input-dir ~/.cache/model-atlas/phase_e_work/ --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Layer 1: Deterministic rules
# ---------------------------------------------------------------------------

# Synonym merges: old → new
SYNONYM_MAP: dict[str, str] = {
    "function-calling": "tool-calling",
    "gguf-available": "gguf-quantized",
}

# If model_id matches regex AND anchor is present → REMOVE anchor
NAME_CONTRADICTION_RULES: list[tuple[re.Pattern, set[str]]] = [
    # Instruct/Chat models are not base models
    (re.compile(r"(?i)(instruct|chat|-it$|-it-)", re.IGNORECASE), {"base-model"}),
    # Dense models are not MoE (no MoE indicator in name)
    # Applied as: if model does NOT match MoE pattern, remove mixture-of-experts
]

# Regex for models that ARE MoE (if name matches, keep mixture-of-experts)
MOE_NAME_PATTERN = re.compile(r"(?i)(moe|MoE|MOE|\d+[bB]-[aA]\d+[bB]|[aA]\d+[bB])")

# Pipeline tag gates: if pipeline_tag matches, REMOVE these anchors
PIPELINE_TAG_GATES: dict[str, set[str]] = {
    "automatic-speech-recognition": {
        "translation", "code-generation", "tool-calling",
        "function-calling", "instruction-following", "reasoning",
    },
    "text-classification": {
        "tool-calling", "function-calling", "code-generation",
        "instruction-following", "reasoning",
    },
    "sentence-similarity": {
        "tool-calling", "function-calling", "instruction-following",
        "reasoning", "code-generation",
    },
    "fill-mask": {
        "tool-calling", "function-calling", "instruction-following",
        "reasoning", "code-generation",
    },
    "token-classification": {
        "tool-calling", "function-calling", "code-generation",
        "reasoning",
    },
    "image-classification": {
        "code-generation", "tool-calling", "function-calling",
        "instruction-following",
    },
    "object-detection": {
        "code-generation", "tool-calling", "function-calling",
        "instruction-following", "reasoning",
    },
    "feature-extraction": {
        "tool-calling", "function-calling", "reasoning",
    },
}

# Benchmark anchors that only make sense for LLMs
LLM_ONLY_ANCHORS = {
    "high-mmlu", "strong-humaneval", "strong-gsm8k",
    "tool-calling", "function-calling", "instruction-following",
    "reasoning", "code-generation", "structured-output",
    "long-context",
}

# Pipeline tags that are NOT LLMs
NON_LLM_TAGS = {
    "automatic-speech-recognition", "text-classification",
    "sentence-similarity", "fill-mask", "token-classification",
    "image-classification", "object-detection", "feature-extraction",
    "image-segmentation", "audio-classification", "depth-estimation",
    "image-to-image", "image-feature-extraction",
}

# Size class definitions: anchor → (min_params_B, max_params_B)
SIZE_CLASS_RANGES: dict[str, tuple[float, float]] = {
    "sub-1b":       (0.0,   0.99),
    "1b-class":     (1.0,   2.99),
    "3b-class":     (3.0,   6.99),
    "7b-class":     (7.0,  12.99),
    "13b-class":   (13.0,  29.99),
    "30b-class":   (30.0,  69.99),
    "70b-class":   (70.0, 199.99),
    "frontier-class": (200.0, 99999.0),
}

ALL_SIZE_CLASSES = set(SIZE_CLASS_RANGES.keys())

# Regex to extract param count from model name
PARAM_COUNT_PATTERN = re.compile(
    r"(?i)(?:^|[-_/])(\d+(?:\.\d+)?)\s*[bB](?:[-_/]|$)"
)

# MoE active param pattern: e.g. "30B-A3B" → active=3B
MOE_ACTIVE_PATTERN = re.compile(
    r"(?i)(\d+(?:\.\d+)?)[bB]-[aA](\d+(?:\.\d+)?)[bB]"
)


def _extract_param_count(model_id: str, existing_metadata: dict) -> float | None:
    """Extract param count in billions from metadata or model name."""
    # Try existing metadata first
    pc = existing_metadata.get("param_count", "")
    if pc:
        try:
            val = float(pc)
            if val > 1e9:
                return val / 1e9
            if val > 0:
                return val
        except (ValueError, TypeError):
            pass

    # Try name regex
    # For MoE models, use total params
    moe_match = MOE_ACTIVE_PATTERN.search(model_id)
    if moe_match:
        return float(moe_match.group(1))

    match = PARAM_COUNT_PATTERN.search(model_id)
    if match:
        return float(match.group(1))

    # Heuristic for known small models
    m = re.search(r"(?i)(\d+)[mM](?:[-_/]|$)", model_id)
    if m:
        return float(m.group(1)) / 1000.0

    return None


def _best_size_class(param_b: float) -> str | None:
    """Return the single best size class for a given param count."""
    for cls, (lo, hi) in SIZE_CLASS_RANGES.items():
        if lo <= param_b <= hi:
            return cls
    return None


@dataclass
class L1Stats:
    """Counters for Layer 1 operations."""
    synonyms_merged: Counter = field(default_factory=Counter)
    name_contradictions_removed: Counter = field(default_factory=Counter)
    moe_bleed_removed: int = 0
    pipeline_tag_gated: Counter = field(default_factory=Counter)
    size_class_deduped: int = 0
    non_llm_anchor_removed: Counter = field(default_factory=Counter)
    models_processed: int = 0
    models_modified: int = 0


def layer1_clean(item: dict, stats: L1Stats) -> dict:
    """Apply deterministic regex rules. Mutates and returns item."""
    stats.models_processed += 1
    model_id = item.get("model_id", "")
    meta = item.get("existing_metadata", {})
    pipeline_tag = meta.get("pipeline_tag", "")
    modified = False

    for bank, bd in item.get("banks", {}).items():
        anchors = bd.get("selected_anchors", [])
        if not anchors:
            continue
        evidence = bd.get("evidence", {})
        original = set(anchors)

        # --- Synonym merges ---
        new_anchors = []
        for a in anchors:
            if a in SYNONYM_MAP:
                replacement = SYNONYM_MAP[a]
                # Avoid duplicates
                if replacement not in new_anchors and replacement not in [x for x in anchors if x != a]:
                    new_anchors.append(replacement)
                    # Migrate evidence
                    if a in evidence:
                        evidence[replacement] = evidence.pop(a)
                    stats.synonyms_merged[a] += 1
                else:
                    # Already has the canonical form, just drop the synonym
                    evidence.pop(a, None)
                    stats.synonyms_merged[a] += 1
            else:
                new_anchors.append(a)
        anchors = new_anchors

        # --- Name-based contradiction: instruct → remove base-model ---
        for pattern, bad_anchors in NAME_CONTRADICTION_RULES:
            if pattern.search(model_id):
                for ba in bad_anchors:
                    if ba in anchors:
                        anchors.remove(ba)
                        evidence.pop(ba, None)
                        stats.name_contradictions_removed[ba] += 1

        # --- MoE bleed: remove mixture-of-experts if name has no MoE signal ---
        if "mixture-of-experts" in anchors and not MOE_NAME_PATTERN.search(model_id):
            anchors.remove("mixture-of-experts")
            evidence.pop("mixture-of-experts", None)
            stats.moe_bleed_removed += 1

        # --- Pipeline tag gates ---
        if pipeline_tag in PIPELINE_TAG_GATES:
            gated = PIPELINE_TAG_GATES[pipeline_tag]
            for ga in list(anchors):
                if ga in gated:
                    anchors.remove(ga)
                    evidence.pop(ga, None)
                    stats.pipeline_tag_gated[ga] += 1

        # --- Non-LLM models shouldn't get LLM-only anchors ---
        if pipeline_tag in NON_LLM_TAGS:
            for la in list(anchors):
                if la in LLM_ONLY_ANCHORS:
                    anchors.remove(la)
                    evidence.pop(la, None)
                    stats.non_llm_anchor_removed[la] += 1

        # --- Size class dedup ---
        size_hits = [a for a in anchors if a in ALL_SIZE_CLASSES]
        if len(size_hits) > 1:
            param_b = _extract_param_count(model_id, meta)
            if param_b is not None:
                best = _best_size_class(param_b)
                if best and best in size_hits:
                    for sh in size_hits:
                        if sh != best:
                            anchors.remove(sh)
                            evidence.pop(sh, None)
                    stats.size_class_deduped += 1
                else:
                    # Keep the closest one
                    assert param_b is not None  # guarded by outer if
                    closest = min(
                        size_hits,
                        key=lambda s: abs(
                            param_b - (SIZE_CLASS_RANGES[s][0] + SIZE_CLASS_RANGES[s][1]) / 2
                        ),
                    )
                    for sh in size_hits:
                        if sh != closest:
                            anchors.remove(sh)
                            evidence.pop(sh, None)
                    stats.size_class_deduped += 1
            else:
                # No param count: keep the larger class (conservative)
                ordered = sorted(
                    size_hits, key=lambda s: SIZE_CLASS_RANGES[s][1], reverse=True
                )
                for sh in ordered[1:]:
                    anchors.remove(sh)
                    evidence.pop(sh, None)
                stats.size_class_deduped += 1

        if set(anchors) != original:
            modified = True
        bd["selected_anchors"] = anchors
        bd["evidence"] = evidence

    if modified:
        stats.models_modified += 1
    return item


# ---------------------------------------------------------------------------
# Layer 2: Structured LLM validation (series-bleed detection)
# ---------------------------------------------------------------------------

# Heuristics to flag items for LLM review
SERIES_BLEED_SUSPECTS = {
    # Anchors that commonly bleed from family descriptions
    "multimodal", "hybrid", "vision-transformer",
    "mixture-of-experts",  # caught by L1, but may have edge cases
}

EVIDENCE_BLEED_PATTERNS = [
    # Evidence that references "the series" or "the family" rather than the specific model
    re.compile(r"(?i)\b(series|family|suite)\b.*\b(offering|includes|provides)\b"),
    re.compile(r"(?i)\b(dense and|and mixture)"),
    re.compile(r"(?i)\bnative vision-language\b"),
]


def _needs_llm_review(item: dict) -> list[tuple[str, str, str]]:
    """Return list of (bank, anchor, evidence) tuples that need LLM validation."""
    suspects = []

    for bank, bd in item.get("banks", {}).items():
        for anchor in bd.get("selected_anchors", []):
            evidence = bd.get("evidence", {}).get(anchor, "")

            # Check if anchor is in the suspect set
            if anchor in SERIES_BLEED_SUSPECTS:
                suspects.append((bank, anchor, evidence))
                continue

            # Check if evidence text has bleed patterns
            for pat in EVIDENCE_BLEED_PATTERNS:
                if pat.search(evidence):
                    suspects.append((bank, anchor, evidence))
                    break

    return suspects


VALIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "anchor": {"type": "string"},
                    "bank": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["KEEP", "REMOVE"]},
                },
                "required": ["anchor", "bank", "verdict"],
            },
        },
    },
    "required": ["verdicts"],
}


def _build_validation_prompt(model_id: str, suspects: list[tuple[str, str, str]]) -> str:
    """Build a prompt for LLM validation of suspect anchors."""
    anchor_block = "\n".join(
        f"  - bank={bank}, anchor=\"{anchor}\", evidence=\"{ev[:200]}\""
        for bank, anchor, ev in suspects
    )

    return f"""You are a strict fact-checker for ML model metadata.

MODEL: {model_id}

The following anchors were extracted from web search results. Some may describe
the model's FAMILY or SERIES rather than THIS SPECIFIC MODEL. Your job is to
decide whether each anchor truly applies to THIS EXACT MODEL.

Rules:
- If the evidence describes the model family/series but not this specific variant, verdict=REMOVE
- If the evidence clearly applies to this specific model, verdict=KEEP
- If the anchor is "multimodal" but the model name has no vision/image/VL indicator, verdict=REMOVE
- If the anchor is "mixture-of-experts" but the model name has no MoE/A*B indicator, verdict=REMOVE
- When in doubt, verdict=REMOVE (precision over recall)

SUSPECT ANCHORS:
{anchor_block}

Return your verdicts as JSON."""


@dataclass
class L2Stats:
    """Counters for Layer 2 operations."""
    models_reviewed: int = 0
    anchors_reviewed: int = 0
    anchors_removed: int = 0
    anchors_kept: int = 0
    llm_errors: int = 0


def layer2_validate(
    item: dict,
    stats: L2Stats,
    ollama_url: str = "http://localhost:11434",
    model: str = "qwen3.5:4b",
) -> dict:
    """Validate suspect anchors via constrained Ollama output. Mutates and returns item."""
    import requests

    suspects = _needs_llm_review(item)
    if not suspects:
        return item

    stats.models_reviewed += 1
    stats.anchors_reviewed += len(suspects)

    prompt = _build_validation_prompt(item["model_id"], suspects)

    try:
        resp = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "format": VALIDATION_SCHEMA,
                "options": {"temperature": 0.1},
                "think": False,
                "stream": False,
            },
            timeout=60,
        )
        resp.raise_for_status()
        result = json.loads(resp.json()["response"])

        verdicts_by_key = {
            (v["bank"], v["anchor"]): v["verdict"]
            for v in result.get("verdicts", [])
        }

        for bank, anchor, _ in suspects:
            verdict = verdicts_by_key.get((bank, anchor), "REMOVE")  # default to REMOVE
            if verdict == "REMOVE":
                bd = item["banks"].get(bank, {})
                anchors = bd.get("selected_anchors", [])
                if anchor in anchors:
                    anchors.remove(anchor)
                    bd.get("evidence", {}).pop(anchor, None)
                    stats.anchors_removed += 1
            else:
                stats.anchors_kept += 1

    except Exception:
        stats.llm_errors += 1
        # On LLM failure, fall back to removing all suspects (precision > recall)
        for bank, anchor, _ in suspects:
            bd = item["banks"].get(bank, {})
            anchors = bd.get("selected_anchors", [])
            if anchor in anchors:
                anchors.remove(anchor)
                bd.get("evidence", {}).pop(anchor, None)
                stats.anchors_removed += 1

    return item


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_result_files(input_dir: Path) -> list[Path]:
    """Find all Phase E result files in a directory."""
    files = []
    for pattern in ["results_*.jsonl", "phase_e_results_*.jsonl"]:
        files.extend(sorted(input_dir.glob(pattern)))
    return files


def process_file(
    input_path: Path,
    output_path: Path | None,
    use_llm: bool,
    dry_run: bool,
    ollama_url: str,
    model: str,
) -> tuple[L1Stats, L2Stats]:
    """Process a single result file through L1 and optionally L2."""
    l1_stats = L1Stats()
    l2_stats = L2Stats()

    items = []
    for line in open(input_path):
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    cleaned = []
    total = len(items)
    for i, item in enumerate(items):
        item = layer1_clean(item, l1_stats)
        if use_llm:
            item = layer2_validate(item, l2_stats, ollama_url, model)
            if l2_stats.models_reviewed > 0 and l2_stats.models_reviewed % 25 == 0:
                print(
                    f"  [{i+1}/{total}] L2: {l2_stats.models_reviewed} reviewed, "
                    f"{l2_stats.anchors_removed} removed, {l2_stats.anchors_kept} kept",
                    flush=True,
                )
        cleaned.append(item)

    if not dry_run and output_path:
        with open(output_path, "w") as f:
            for item in cleaned:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return l1_stats, l2_stats


def _print_stats(filename: str, l1: L1Stats, l2: L2Stats) -> None:
    """Print stats for one file."""
    print(f"\n{'='*60}")
    print(f"  {filename}")
    print(f"{'='*60}")
    print(f"  Models: {l1.models_processed} processed, {l1.models_modified} modified")
    print("\n  Layer 1 (deterministic):")
    if l1.synonyms_merged:
        print(f"    Synonyms merged: {dict(l1.synonyms_merged)}")
    if l1.name_contradictions_removed:
        print(f"    Name contradictions: {dict(l1.name_contradictions_removed)}")
    if l1.moe_bleed_removed:
        print(f"    MoE bleed removed: {l1.moe_bleed_removed}")
    if l1.pipeline_tag_gated:
        print(f"    Pipeline-tag gated: {dict(l1.pipeline_tag_gated)}")
    if l1.non_llm_anchor_removed:
        print(f"    Non-LLM anchor removed: {dict(l1.non_llm_anchor_removed)}")
    if l1.size_class_deduped:
        print(f"    Size class deduped: {l1.size_class_deduped}")

    if l2.models_reviewed:
        print("\n  Layer 2 (LLM validation):")
        print(f"    Models reviewed: {l2.models_reviewed}")
        print(f"    Anchors reviewed: {l2.anchors_reviewed}")
        print(f"    Kept: {l2.anchors_kept}, Removed: {l2.anchors_removed}")
        if l2.llm_errors:
            print(f"    LLM errors (fallback to REMOVE): {l2.llm_errors}")


def main():
    parser = argparse.ArgumentParser(description="Phase E post-processor")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="Single input JSONL file")
    group.add_argument("--input-dir", type=Path, help="Directory with result files")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: same dir, _cleaned suffix)")
    parser.add_argument("--llm", action="store_true", help="Enable Layer 2 LLM validation")
    parser.add_argument("--dry-run", action="store_true", help="Report only, no output files")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama API URL")
    parser.add_argument("--model", default="qwen3.5:4b", help="Ollama model for L2 validation")
    args = parser.parse_args()

    if args.input:
        files = [args.input]
    else:
        files = _find_result_files(args.input_dir)
        if not files:
            print(f"No result files found in {args.input_dir}", file=sys.stderr)
            sys.exit(1)

    total_l1 = L1Stats()
    total_l2 = L2Stats()

    for f in files:
        if args.output_dir:
            out = args.output_dir / f"{f.stem}_cleaned.jsonl"
        else:
            out = f.parent / f"{f.stem}_cleaned.jsonl"

        l1, l2 = process_file(f, out, args.llm, args.dry_run, args.ollama_url, args.model)
        _print_stats(f.name, l1, l2)

        # Accumulate totals
        total_l1.models_processed += l1.models_processed
        total_l1.models_modified += l1.models_modified
        total_l1.synonyms_merged += l1.synonyms_merged
        total_l1.name_contradictions_removed += l1.name_contradictions_removed
        total_l1.moe_bleed_removed += l1.moe_bleed_removed
        total_l1.pipeline_tag_gated += l1.pipeline_tag_gated
        total_l1.non_llm_anchor_removed += l1.non_llm_anchor_removed
        total_l1.size_class_deduped += l1.size_class_deduped
        total_l2.models_reviewed += l2.models_reviewed
        total_l2.anchors_reviewed += l2.anchors_reviewed
        total_l2.anchors_removed += l2.anchors_removed
        total_l2.anchors_kept += l2.anchors_kept
        total_l2.llm_errors += l2.llm_errors

    if len(files) > 1:
        _print_stats("TOTAL (all files)", total_l1, total_l2)

    if not args.dry_run and not args.input:
        print(f"\nCleaned files written to: {args.output_dir or files[0].parent}")


if __name__ == "__main__":
    main()
