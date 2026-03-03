"""Standalone Phase C3 quality gate worker.

Self-contained single-file script with zero ModelAtlas imports.
Dependencies: openai, json, argparse. Can be scp'd to any machine
with Python + `pip install openai`.

Usage:
    python phase_c3_worker.py --input quality_gate.jsonl --output results_c3.jsonl
    python phase_c3_worker.py --input quality_gate.jsonl --output results_c3.jsonl \
        --model qwen2.5:3b --url http://192.168.50.17:11434/v1
"""

from __future__ import annotations

import argparse
import json
import signal
import sys

_shutdown = False


def _handle_signal(signum: int, frame: object) -> None:
    global _shutdown
    print(f"Received signal {signum}, finishing current model...", file=sys.stderr)
    _shutdown = True


_VALID_FLAGS = frozenset({
    "generic", "hallucinated", "truncated", "repetitive",
})


def _parse_and_validate(text: str) -> dict:
    """Parse and validate quality gate JSON output."""
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")

    specificity = data.get("specificity")
    coherence = data.get("coherence")
    artifacts = data.get("artifacts")
    flags = data.get("flags", [])

    for name, value in [("specificity", specificity), ("coherence", coherence), ("artifacts", artifacts)]:
        if not isinstance(value, (int, float)):
            raise ValueError(f"'{name}' must be int or float, got {type(value).__name__}")
        if not (0 <= value <= 3):
            raise ValueError(f"'{name}' must be 0-3, got {value}")

    if not isinstance(flags, list):
        raise ValueError("'flags' must be a list")

    # Filter to only known flag values
    flags = [f for f in flags if isinstance(f, str) and f.strip().lower() in _VALID_FLAGS]

    specificity = int(specificity)
    coherence = int(coherence)
    artifacts = int(artifacts)
    quality_score = (specificity + coherence + artifacts) / 9.0

    return {
        "quality_score": round(quality_score, 4),
        "specificity": specificity,
        "coherence": coherence,
        "artifacts": artifacts,
        "flags": flags,
    }


def _load_skip_set(output_path: str) -> set[str]:
    """Read existing output file and return set of already-processed model_ids."""
    skip = set()
    try:
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    mid = item.get("model_id", "")
                    if mid:
                        skip.add(mid)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return skip


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Phase C3 quality gate worker")
    parser.add_argument("--input", required=True, help="Input quality gate JSONL file")
    parser.add_argument("--output", required=True, help="Output results JSONL file")
    parser.add_argument("--model", default="qwen2.5:3b", help="Ollama model name")
    parser.add_argument(
        "--url", default="http://localhost:11434/v1", help="Ollama API base URL"
    )
    parser.add_argument("--resume", action="store_true", help="Skip already-processed model_ids")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    from openai import OpenAI

    client = OpenAI(base_url=args.url, api_key="ollama")

    skip_set: set[str] = set()
    if args.resume:
        skip_set = _load_skip_set(args.output)
        if skip_set:
            print(f"Resume: skipping {len(skip_set)} already-processed models", file=sys.stderr)

    file_mode = "a" if args.resume else "w"
    count = 0
    skipped = 0
    errors = 0

    with open(args.input) as fin, open(args.output, file_mode) as fout:
        for line in fin:
            if _shutdown:
                break

            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            model_id = item.get("model_id", "")
            prompt = item.get("prompt", "")

            if model_id in skip_set:
                skipped += 1
                continue

            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                )
                text = response.choices[0].message.content or ""
                result = _parse_and_validate(text)
                out = json.dumps({"model_id": model_id, **result})
            except Exception as e:
                out = json.dumps({"model_id": model_id, "error": str(e)})
                errors += 1

            fout.write(out + "\n")
            fout.flush()
            count += 1

            if count % 10 == 0:
                print(f"Progress: {count} processed ({errors} errors)", file=sys.stderr)

    print(f"Done: {count} processed, {errors} errors, {skipped} skipped", file=sys.stderr)


if __name__ == "__main__":
    main()
