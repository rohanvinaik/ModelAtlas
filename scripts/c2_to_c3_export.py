"""Convert C2 results JSONL directly to C3 quality gate input JSONL.

Skips the DB merge step — builds C3 prompts from C2 output fields.

Usage:
    python c2_to_c3_export.py --input results_1.jsonl --output c3_shard_1.jsonl
"""

from __future__ import annotations

import argparse
import json

_QUALITY_GATE_TEMPLATE = """You are a blind quality reviewer for ML model summaries. Given ONLY a model ID, its summary, and its tags, score the summary on three axes (0-3 each):

- specificity: Does the summary mention concrete distinguishing details (architecture, dataset, size, technique)? 0=generic/boilerplate, 3=highly specific and informative.
- coherence: Is the summary well-formed, grammatical, and internally consistent? 0=garbled/contradictory, 3=clear and professional.
- artifacts: Does the summary contain LLM artifacts (repetition, hallucinated URLs, prompt leakage, filler phrases)? 0=severe artifacts, 3=clean.

Also list any flags (empty list if none): "generic" (could apply to any model), "hallucinated" (claims not supported by tags), "truncated" (sentence cut off), "repetitive" (repeated phrases).

Model: {model_id}
Summary: {summary}
Tags: {tags}

Respond with valid JSON containing keys: "specificity" (int 0-3), "coherence" (int 0-3), "artifacts" (int 0-3), "flags" (array of strings, empty if none)."""


def main():
    parser = argparse.ArgumentParser(description="Convert C2 results to C3 input")
    parser.add_argument("--input", required=True, help="C2 results JSONL")
    parser.add_argument("--output", required=True, help="C3 input JSONL")
    args = parser.parse_args()

    count = 0
    skipped = 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            model_id = item.get("model_id", "")
            summary = item.get("summary", "")
            if "error" in item or not summary:
                skipped += 1
                continue

            tags = item.get("extra_anchors", [])
            tag_str = ", ".join(tags[:15]) if tags else "none"

            prompt = _QUALITY_GATE_TEMPLATE.format(
                model_id=model_id,
                summary=summary,
                tags=tag_str,
            )

            fout.write(json.dumps({"model_id": model_id, "prompt": prompt}) + "\n")
            count += 1

    print(f"Exported {count} models for C3 ({skipped} skipped)")


if __name__ == "__main__":
    main()
