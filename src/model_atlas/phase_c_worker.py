"""Standalone Phase C vibe extraction worker.

Self-contained single-file script with zero ModelAtlas imports.
Dependencies: openai, json, argparse. Can be scp'd to any machine
with Python + `pip install openai`.

Usage:
    python phase_c_worker.py --input shard.jsonl --output results.jsonl
    python phase_c_worker.py --input shard.jsonl --output results.jsonl \
        --model qwen2.5:3b --url http://localhost:11434/v1
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


def _parse_and_validate(text: str, valid_anchors: set[str] | None = None) -> dict:
    """Parse and validate vibe JSON output.

    When valid_anchors is provided, returned anchors are filtered to only
    those in the dictionary. This is the primary quality gate — dictionary
    membership replaces free-form pattern checks.
    """
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")
    summary = data.get("summary", "")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("Missing or empty 'summary'")
    raw_anchors = data.get("selected_anchors") or data.get("extra_anchors") or []
    if not isinstance(raw_anchors, list):
        raise ValueError("'selected_anchors' must be a list")
    cleaned = []
    for a in raw_anchors:
        if not isinstance(a, str):
            continue
        a = a.strip().lower()
        if len(a) < 3:
            continue
        if valid_anchors is not None and a not in valid_anchors:
            continue
        cleaned.append(a)
    return {"summary": summary.strip(), "selected_anchors": cleaned[:5]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Phase C vibe worker")
    parser.add_argument("--input", required=True, help="Input shard JSONL file")
    parser.add_argument("--output", required=True, help="Output results JSONL file")
    parser.add_argument("--model", default="qwen2.5:3b", help="Ollama model name")
    parser.add_argument(
        "--url", default="http://localhost:11434/v1", help="Ollama API base URL"
    )
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    from openai import OpenAI

    client = OpenAI(base_url=args.url, api_key="ollama")

    count = 0
    errors = 0

    with open(args.input) as fin, open(args.output, "w") as fout:
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
            valid_anchors_list = item.get("valid_anchors")
            valid_anchors = set(valid_anchors_list) if valid_anchors_list else None

            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                )
                text = response.choices[0].message.content or ""
                result = _parse_and_validate(text, valid_anchors=valid_anchors)
                out = json.dumps({"model_id": model_id, **result})
            except Exception as e:
                out = json.dumps({"model_id": model_id, "error": str(e)})
                errors += 1

            fout.write(out + "\n")
            fout.flush()
            count += 1

            if count % 10 == 0:
                print(f"Progress: {count} processed ({errors} errors)", file=sys.stderr)

    print(f"Done: {count} processed, {errors} errors", file=sys.stderr)


if __name__ == "__main__":
    main()
