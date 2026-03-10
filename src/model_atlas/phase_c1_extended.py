"""Extended C1: Smol-Hub-tldr across all HuggingFace.

Three-tier processing: newest HF API models, librarian-bots 975K corpus,
then remaining models. Card text is transient — never stored on disk.

Supports --from-ids to process a specific set of model IDs (e.g. from
export_c1 output) instead of streaming from HF API tiers.

Usage:
    python phase_c1_extended.py --output results_c1_ext.jsonl [--resume]
    python phase_c1_extended.py --output results_c1_ext.jsonl --tier 2 --resume
    python phase_c1_extended.py --output results_c1_ext.jsonl --from-ids models_for_c1.jsonl
    python phase_c1_extended.py --output results_c1_ext.jsonl --chunk-size 250 --max-models 1000
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import sys
import time
from dataclasses import dataclass

MODEL_NAME = "davanstrien/Smol-Hub-tldr"
MAX_LENGTH = 2048
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.4
LIBRARIAN_BOTS_DATASET = "librarian-bots/model_cards_with_metadata"
CHUNK_SLEEP_SECONDS = 1

_shutdown = False


def _handle_signal(signum: int, frame: object) -> None:
    global _shutdown
    print(f"Received signal {signum}, finishing current model...", file=sys.stderr)
    _shutdown = True


# ---------------------------------------------------------------------------
# Card text cleaning
# ---------------------------------------------------------------------------

_RE_HTML_TAGS = re.compile(r"<[^>]*>?")
_RE_HTML_ENTITIES = re.compile(r"&#?\w+;")
_RE_BADGE_LINKED = re.compile(r"\[!\[[^\]]*\]\([^\)]*\)\]\([^\)]*\)")
_RE_BADGE_INLINE = re.compile(r"!\[[^\]]*\]\([^\)]*\)")
_RE_URLS = re.compile(r"https?://[^\s\)]+")
_RE_FENCED_CODE = re.compile(r"```[\s\S]*?```")
_RE_WHITESPACE = re.compile(r"\s+")


def clean_card_text(text: str) -> str:
    """Strip HTML, badges, URLs, code blocks, collapse whitespace, truncate."""
    text = _RE_HTML_TAGS.sub("", text)
    text = _RE_HTML_ENTITIES.sub(" ", text)
    text = _RE_BADGE_LINKED.sub("", text)
    text = _RE_BADGE_INLINE.sub("", text)
    text = _RE_URLS.sub("", text)
    text = _RE_FENCED_CODE.sub("", text)
    text = _RE_WHITESPACE.sub(" ", text).strip()
    return text[:2000]


# ---------------------------------------------------------------------------
# Resume / skip set
# ---------------------------------------------------------------------------


def _load_skip_set(output_path: str) -> set[str]:
    """Read existing output file and return set of already-processed model_ids."""
    skip: set[str] = set()
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


# ---------------------------------------------------------------------------
# Librarian-bots ID set (loaded once for Tier 1 dedup)
# ---------------------------------------------------------------------------


def _load_librarian_bots_ids() -> set[str]:
    """Stream librarian-bots dataset and collect all model_ids."""
    from datasets import load_dataset

    print("Loading librarian-bots model_id set (streaming)...", file=sys.stderr)
    ids: set[str] = set()
    ds = load_dataset(LIBRARIAN_BOTS_DATASET, split="train", streaming=True)
    for row in ds:
        mid = row.get("modelId") or row.get("model_id") or ""
        if mid:
            ids.add(mid)
    print(f"Librarian-bots set: {len(ids)} model_ids", file=sys.stderr)
    return ids


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model(model_name: str):
    """Load Smol-Hub-tldr model and tokenizer, return (model, tokenizer, device)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model {model_name}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}", file=sys.stderr)
    return model, tokenizer, device


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------


def _summarize(card_text: str, model, tokenizer, device: str) -> str:
    """Clean card text, run through Smol-Hub-tldr, return summary string."""
    import torch

    cleaned = clean_card_text(card_text)
    if not cleaned.strip():
        return ""

    prompt = f"<MODEL_CARD>{cleaned}</MODEL_CARD>"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        truncation=True,
    ).to(device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
        )

    new_tokens = outputs[0][input_length:]
    summary = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return summary


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def _write_result(
    fout, model_id: str, tier: int, summary: str | None = None, error: str | None = None
) -> None:
    """Write a single JSONL result line."""
    if error is not None:
        record = {"model_id": model_id, "error": error, "tier": tier}
    else:
        record = {"model_id": model_id, "smol_summary": summary or "", "tier": tier}
    fout.write(json.dumps(record) + "\n")
    fout.flush()


# ---------------------------------------------------------------------------
# Shared processing context and helpers
# ---------------------------------------------------------------------------


@dataclass
class _ProcessCtx:
    """Bundles shared state for model processing across all tiers."""

    model: object
    tokenizer: object
    device: str
    fout: object
    skip_set: set[str]
    max_models: int
    count: int = 0
    errors: int = 0
    tier_count: int = 0

    def should_stop(self) -> bool:
        """Check if we've hit the model limit or received a shutdown signal."""
        if _shutdown:
            return True
        return self.max_models > 0 and self.count >= self.max_models

    def process_one_hf(self, mid: str, tier: int) -> None:
        """Fetch card from HF, summarize, write result."""
        from huggingface_hub import ModelCard

        try:
            card = ModelCard.load(mid)
            card_text = card.text if card.text else ""
            summary = _summarize(card_text, self.model, self.tokenizer, self.device)
            _write_result(self.fout, mid, tier=tier, summary=summary)
        except Exception as e:
            _write_result(self.fout, mid, tier=tier, error=str(e))
            self.errors += 1
        self.skip_set.add(mid)
        self.count += 1
        self.tier_count += 1

    def log_progress(self, label: str) -> None:
        """Print progress every 100 models."""
        if self.tier_count % 100 == 0:
            print(
                f"{label} progress: {self.tier_count} processed ({self.errors} errors)",
                file=sys.stderr,
            )


def _extract_model_id(info) -> str:
    """Get model_id string from an HF API model info object."""
    return info.id if hasattr(info, "id") else str(info)


# ---------------------------------------------------------------------------
# Chunked HF API processing (shared by Tier 1 and Tier 3)
# ---------------------------------------------------------------------------


def _process_chunk(
    ctx: _ProcessCtx,
    chunk: list,
    tier: int,
    label: str,
    extra_skip: set[str] | None = None,
    cutoff_check=None,
) -> bool:
    """Process a chunk of HF API model infos. Returns True to stop early."""
    for info in chunk:
        if ctx.should_stop():
            return True

        mid = _extract_model_id(info)
        if mid in ctx.skip_set:
            continue
        if extra_skip and mid in extra_skip:
            continue
        if cutoff_check and cutoff_check(info):
            return True

        ctx.process_one_hf(mid, tier)
        ctx.log_progress(label)

    return False


def _run_chunked_api(
    ctx: _ProcessCtx,
    tier: int,
    label: str,
    chunk_size: int,
    extra_skip: set[str] | None = None,
    cutoff_check=None,
) -> tuple[int, int]:
    """Iterate HF API in chunks, process each model, sleep between chunks."""
    from huggingface_hub import HfApi

    print(f"=== {label} ===", file=sys.stderr)
    api = HfApi()
    ctx.tier_count = 0
    chunk: list = []

    for model_info in api.list_models(sort="createdAt"):
        if ctx.should_stop():
            break
        chunk.append(model_info)
        if len(chunk) < chunk_size:
            continue

        stop = _process_chunk(
            ctx, chunk, tier, label, extra_skip=extra_skip, cutoff_check=cutoff_check
        )
        chunk.clear()
        if stop:
            break
        time.sleep(CHUNK_SLEEP_SECONDS)

    # Remaining partial chunk
    if chunk and not ctx.should_stop():
        _process_chunk(
            ctx, chunk, tier, label, extra_skip=extra_skip, cutoff_check=cutoff_check
        )

    print(
        f"{label} done: {ctx.tier_count} processed ({ctx.errors} errors)",
        file=sys.stderr,
    )
    return ctx.count, ctx.errors


# ---------------------------------------------------------------------------
# From-IDs mode: process specific model IDs from a JSONL file
# ---------------------------------------------------------------------------


def _iter_ids_from_file(ids_file: str) -> list[str]:
    """Read model IDs from a JSONL file, skipping malformed lines."""
    ids: list[str] = []
    with open(ids_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            mid = item.get("model_id", "")
            if mid:
                ids.append(mid)
    return ids


def _run_from_ids(ctx: _ProcessCtx, ids_file: str) -> tuple[int, int]:
    """Process specific model IDs by fetching card text from HF API."""
    print(f"=== From-IDs mode: processing models from {ids_file} ===", file=sys.stderr)
    ctx.tier_count = 0

    for mid in _iter_ids_from_file(ids_file):
        if ctx.should_stop():
            break
        if mid in ctx.skip_set:
            continue

        ctx.process_one_hf(mid, tier=0)
        ctx.log_progress("From-IDs")

    print(
        f"From-IDs done: {ctx.tier_count} processed ({ctx.errors} errors)",
        file=sys.stderr,
    )
    return ctx.count, ctx.errors


# ---------------------------------------------------------------------------
# Tier 1: Newest HF API models not in librarian-bots
# ---------------------------------------------------------------------------


def _make_tier1_cutoff_check(cutoff):
    """Build a cutoff checker for Tier 1 date filtering."""

    def check(info) -> bool:
        created = getattr(info, "created_at", None)
        if created is not None and created < cutoff:
            print(
                f"Tier 1: Hit cutoff date ({cutoff.date()}), stopping.",
                file=sys.stderr,
            )
            return True
        return False

    return check


def _run_tier1(
    ctx: _ProcessCtx, librarian_ids: set[str], chunk_size: int
) -> tuple[int, int]:
    """Process newest models from HF API that are not in librarian-bots."""
    from datetime import datetime, timedelta, timezone

    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    return _run_chunked_api(
        ctx,
        tier=1,
        label="Tier 1: Newest HF API models (not in librarian-bots)",
        chunk_size=chunk_size,
        extra_skip=librarian_ids,
        cutoff_check=_make_tier1_cutoff_check(cutoff),
    )


# ---------------------------------------------------------------------------
# Tier 2: librarian-bots/model_cards_with_metadata (975K, by recency)
# ---------------------------------------------------------------------------


def _run_tier2(ctx: _ProcessCtx) -> tuple[int, int]:
    """Process librarian-bots corpus via streaming dataset."""
    from datasets import load_dataset

    print(
        "=== Tier 2: librarian-bots/model_cards_with_metadata (streaming) ===",
        file=sys.stderr,
    )
    ctx.tier_count = 0
    ds = load_dataset(LIBRARIAN_BOTS_DATASET, split="train", streaming=True)

    for row in ds:
        if ctx.should_stop():
            break

        mid = row.get("modelId") or row.get("model_id") or ""
        if not mid or mid in ctx.skip_set:
            continue

        card_text = row.get("card") or row.get("text") or row.get("card_text") or ""

        try:
            summary = _summarize(card_text, ctx.model, ctx.tokenizer, ctx.device)
            _write_result(ctx.fout, mid, tier=2, summary=summary)
        except Exception as e:
            _write_result(ctx.fout, mid, tier=2, error=str(e))
            ctx.errors += 1

        ctx.skip_set.add(mid)
        ctx.count += 1
        ctx.tier_count += 1
        ctx.log_progress("Tier 2")

    print(
        f"Tier 2 done: {ctx.tier_count} processed ({ctx.errors} errors)",
        file=sys.stderr,
    )
    return ctx.count, ctx.errors


# ---------------------------------------------------------------------------
# Tier 3: Remaining models from HF API
# ---------------------------------------------------------------------------


def _run_tier3(ctx: _ProcessCtx, chunk_size: int) -> tuple[int, int]:
    """Scan HF API for any model_ids not yet processed."""
    return _run_chunked_api(
        ctx,
        tier=3,
        label="Tier 3: Remaining HF API models",
        chunk_size=chunk_size,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extended C1: Smol-Hub-tldr across all HuggingFace"
    )
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument(
        "--resume", action="store_true", help="Skip already-processed models"
    )
    parser.add_argument("--chunk-size", type=int, default=250, help="HF API chunk size")
    parser.add_argument(
        "--max-models",
        type=int,
        default=0,
        help="Max models to process (0 = unlimited)",
    )
    parser.add_argument(
        "--tier",
        default="all",
        choices=["1", "2", "3", "all"],
        help="Which tier to run: 1, 2, 3, or all",
    )
    parser.add_argument(
        "--from-ids",
        help="JSONL file of model IDs to process (skips tier system)",
    )
    parser.add_argument(
        "--model", default=MODEL_NAME, help="Model name for summarization"
    )
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # Resume support
    skip_set: set[str] = set()
    if args.resume:
        skip_set = _load_skip_set(args.output)
        print(f"Resume: {len(skip_set)} model_ids already processed", file=sys.stderr)

    # Load model
    lm, tokenizer, device = _load_model(args.model)
    open_mode = "a" if args.resume else "w"

    with open(args.output, open_mode) as fout:
        ctx = _ProcessCtx(
            model=lm,
            tokenizer=tokenizer,
            device=device,
            fout=fout,
            skip_set=skip_set,
            max_models=args.max_models,
        )

        if args.from_ids:
            _run_from_ids(ctx, args.from_ids)
        else:
            run_tiers = [args.tier] if args.tier != "all" else ["1", "2", "3"]
            librarian_ids: set[str] = set()
            if "1" in run_tiers:
                librarian_ids = _load_librarian_bots_ids()

            if "1" in run_tiers and not ctx.should_stop():
                _run_tier1(ctx, librarian_ids, args.chunk_size)

            if "2" in run_tiers and not ctx.should_stop():
                _run_tier2(ctx)

            if "3" in run_tiers and not ctx.should_stop():
                _run_tier3(ctx, args.chunk_size)

    print(
        f"All done: {ctx.count} processed, {ctx.errors} errors, "
        f"{len(skip_set)} total unique model_ids seen",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
