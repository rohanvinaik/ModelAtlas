"""Standalone Phase C1 Smol-Hub-tldr summarization worker.

Self-contained single-file script with zero ModelAtlas imports.
Dependencies: transformers, torch, argparse, json. Can be scp'd to any machine
with Python + `pip install transformers torch`.

Usage:
    python phase_c1_worker.py --input cards.jsonl --output results_c1.jsonl
    python phase_c1_worker.py --input cards.jsonl --output results_c1.jsonl --resume
"""

from __future__ import annotations

import argparse
import json
import signal
import sys

MODEL_NAME = "davanstrien/Smol-Hub-tldr"
MAX_LENGTH = 2048
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.4

_shutdown = False


def _handle_signal(signum: int, frame: object) -> None:
    global _shutdown
    print(f"Received signal {signum}, finishing current model...", file=sys.stderr)
    _shutdown = True


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


def _run_inference(args, card_text, device, model, tokenizer, torch):
    prompt = f"<MODEL_CARD>{card_text}</MODEL_CARD>"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=args.max_length,
        truncation=True,
    ).to(device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=TEMPERATURE,
            do_sample=True,
        )

    new_tokens = outputs[0][input_length:]
    summary = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone Phase C1 Smol-Hub-tldr worker"
    )
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output results JSONL file")
    parser.add_argument("--model", default=MODEL_NAME, help="Model name")
    parser.add_argument(
        "--resume", action="store_true", help="Skip already-processed model_ids"
    )
    parser.add_argument(
        "--max-length", type=int, default=MAX_LENGTH, help="Max input length"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Max new tokens to generate",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    skip_set: set[str] = set()
    if args.resume:
        skip_set = _load_skip_set(args.output)
        print(f"Resume: {len(skip_set)} model_ids already processed", file=sys.stderr)

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f"Loading model {args.model}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}", file=sys.stderr)

    count = 0
    skipped = 0
    errors = 0

    open_mode = "a" if args.resume else "w"
    with open(args.input) as fin, open(args.output, open_mode) as fout:
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
            card_text = item.get("card_text", "")

            if model_id in skip_set:
                skipped += 1
                continue

            try:
                summary = _run_inference(args, card_text, device, model, tokenizer, torch)
                out = json.dumps({"model_id": model_id, "smol_summary": summary})
            except Exception as e:
                out = json.dumps({"model_id": model_id, "error": str(e)})
                errors += 1

            fout.write(out + "\n")
            fout.flush()
            count += 1

            if count % 10 == 0:
                print(f"Progress: {count} processed ({errors} errors)", file=sys.stderr)

    print(
        f"Done: {count} processed, {errors} errors, {skipped} skipped",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
