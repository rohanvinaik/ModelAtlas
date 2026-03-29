"""Standalone Phase E web enrichment worker.

Self-contained single-file script with zero ModelAtlas imports.
Searches the open web via DDG for information NOT in HuggingFace model
cards (benchmarks, comparisons, usage reports), then extracts structured
bank anchors via local Ollama (native API, think=false).

DDG-only by design: Phase A/B/C already extracted from HF model cards.
Phase E finds what the cards don't contain.

Zero external dependencies — pure stdlib (urllib, json, argparse).
Can be scp'd to any machine with Python 3.10+.

Usage:
    python phase_e_worker.py --input shard.jsonl --output results.jsonl
    python phase_e_worker.py --input shard.jsonl --output results.jsonl \
        --model qwen3.5:4b --url http://localhost:11434 \
        --banks CAPABILITY,DOMAIN,QUALITY --resume
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import os
import re
import signal
import sys
import time
import urllib.parse
import urllib.request

_shutdown = False

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _handle_signal(signum: int, frame: object) -> None:
    global _shutdown
    print(f"Received signal {signum}, finishing current model...", file=sys.stderr)
    _shutdown = True


# ---------------------------------------------------------------------------
# Stage 1: Web Search (SearXNG primary, DDG fallback)
# ---------------------------------------------------------------------------

def searxng_search(
    query: str, searxng_url: str, max_results: int = 5, timeout: int = 15
) -> list[dict]:
    """Search via local SearXNG instance. Returns [{url, title, snippet}]."""
    params = urllib.parse.urlencode({
        "q": query, "format": "json", "categories": "general",
    })
    url = f"{searxng_url}/search?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        data = json.loads(resp.read())
    except Exception as e:
        print(f"  SearXNG search failed: {e}", file=sys.stderr)
        return []

    results = []
    seen: set[str] = set()
    for r in data.get("results", [])[:max_results]:
        u = r.get("url", "")
        if not u or u in seen:
            continue
        seen.add(u)
        results.append({
            "url": u,
            "title": r.get("title", ""),
            "snippet": r.get("content", ""),
        })
    return results


def ddg_search(query: str, max_results: int = 5, timeout: int = 15) -> list[dict]:
    """Search DuckDuckGo via HTML endpoint (fallback). Returns [{url, title, snippet}]."""
    data = urllib.parse.urlencode({"q": query}).encode()
    req = urllib.request.Request(
        "https://html.duckduckgo.com/html/",
        data=data,
        headers={"User-Agent": _USER_AGENT},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        body = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  DDG search failed: {e}", file=sys.stderr)
        return []

    link_pattern = re.compile(
        r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>', re.DOTALL
    )
    snippet_pattern = re.compile(
        r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL
    )
    links = link_pattern.findall(body)
    snippets = snippet_pattern.findall(body)

    if not links and ("Please verify" in body or "bot" in body.lower()):
        print("  DDG CAPTCHA detected", file=sys.stderr)
        return []

    results = []
    seen_urls: set[str] = set()
    for i, (raw_url, raw_title) in enumerate(links[:max_results]):
        u = _extract_ddg_url(raw_url)
        if not u or u in seen_urls:
            continue
        seen_urls.add(u)
        title = re.sub(r"<[^>]+>", "", html_mod.unescape(raw_title)).strip()
        snippet = ""
        if i < len(snippets):
            snippet = re.sub(r"<[^>]+>", "", html_mod.unescape(snippets[i])).strip()
        results.append({"url": u, "title": title, "snippet": snippet})
    return results


def _extract_ddg_url(raw: str) -> str | None:
    """Extract actual URL from DDG redirect."""
    match = re.search(r"uddg=([^&]+)", raw)
    if match:
        return urllib.parse.unquote(match.group(1))
    return raw if raw.startswith("http") else None


def web_search(
    query: str, searxng_url: str | None, max_results: int = 5, timeout: int = 15
) -> list[dict]:
    """Search with fallback cascade: SearXNG → DDG."""
    if searxng_url:
        results = searxng_search(query, searxng_url, max_results, timeout)
        if results:
            return results
        # SearXNG failed, fall through to DDG
    return ddg_search(query, max_results, timeout)


# ---------------------------------------------------------------------------
# Stage 2: Page Fetch
# ---------------------------------------------------------------------------

def fetch_page(url: str, max_chars: int = 50000, timeout: int = 15) -> str | None:
    """Fetch URL, strip HTML to readable text."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        ctype = resp.headers.get("Content-Type", "")
        if "text" not in ctype and "html" not in ctype and "json" not in ctype:
            return None
        raw = resp.read(max_chars).decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  Fetch failed ({url[:60]}): {e}", file=sys.stderr)
        return None

    # Strip script/style/nav/header/footer
    for tag in ("script", "style", "nav", "header", "footer"):
        raw = re.sub(
            rf"<{tag}[^>]*>.*?</{tag}>", " ", raw, flags=re.DOTALL | re.IGNORECASE
        )
    # Block elements → newlines
    raw = re.sub(
        r"<(?:p|div|br|h[1-6]|li|tr)[^>]*>", "\n", raw, flags=re.IGNORECASE
    )
    # Strip remaining tags
    raw = re.sub(r"<[^>]+>", " ", raw)
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", html_mod.unescape(raw)).strip()
    # Collapse runs of spaces on each line
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


# ---------------------------------------------------------------------------
# Stage 3: Content Aggregation + Relevance Scoring
# ---------------------------------------------------------------------------

_EVALUATIVE_TERMS = [
    "excellent", "outstanding", "poor", "mediocre", "benchmark",
    "compared to", "better than", "worse than", "outperforms",
    "evaluation", "results", "score", "accuracy", "performance",
    "beats", "state-of-the-art", "SOTA", "leaderboard",
    "fine-tuned", "trained on", "capabilities", "limitations",
    "mmlu", "humaneval", "gsm8k", "hellaswag", "arc",
]


def _score_paragraph(para: str, model_terms: set[str]) -> float:
    """Score a paragraph by relevance to the model and evaluative density."""
    lower = para.lower()
    model_hits = sum(1 for t in model_terms if t in lower)
    eval_hits = sum(1 for t in _EVALUATIVE_TERMS if t in lower)
    length_bonus = min(len(para) / 500.0, 1.0)  # prefer substantive paragraphs
    return model_hits * 0.5 + eval_hits * 0.3 + length_bonus * 0.2


def aggregate_content(
    pages: list[str],
    model_id: str,
    max_chars: int = 8000,
) -> str:
    """Select best paragraphs from fetched pages up to context budget."""
    # Build relevance terms from model_id
    model_terms: set[str] = set()
    for part in model_id.lower().replace("/", " ").replace("-", " ").split():
        if len(part) > 2:
            model_terms.add(part)

    # Split all pages into paragraphs and score
    scored: list[tuple[float, str]] = []
    for page_text in pages:
        if not page_text:
            continue
        paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
        for para in paragraphs:
            if len(para) < 30:  # skip tiny fragments
                continue
            score = _score_paragraph(para, model_terms)
            scored.append((score, para))

    scored.sort(key=lambda x: -x[0])

    # Take top paragraphs up to budget
    selected: list[str] = []
    total = 0
    for _, para in scored:
        if total + len(para) > max_chars:
            break
        selected.append(para)
        total += len(para)

    return "\n\n".join(selected)


# ---------------------------------------------------------------------------
# Stage 4: Bank-Specific LLM Extraction
# ---------------------------------------------------------------------------

_BANK_DESCRIPTIONS = {
    "ARCHITECTURE": "model architecture and structure (transformer, encoder-only, decoder-only, MoE, etc.)",
    "CAPABILITY": "what the model can do (code generation, reasoning, chat, tool-calling, etc.)",
    "EFFICIENCY": "model size, quantization, and computational requirements",
    "COMPATIBILITY": "formats, frameworks, and hardware compatibility (GGUF, vLLM, Apple Silicon, etc.)",
    "LINEAGE": "model family, base model, training lineage (Llama, Mistral, fine-tune, merge, etc.)",
    "DOMAIN": "domain specialization (code, medical, legal, finance, multilingual, etc.)",
    "QUALITY": "reputation, benchmark performance, community reception",
    "TRAINING": "training methodology (SFT, RLHF, DPO, LoRA, distillation, etc.)",
}


def _build_extraction_prompt(
    model_id: str,
    bank: str,
    valid_anchors: list[str],
    existing_anchors: list[str],
    existing_metadata: dict,
    web_content: str,
) -> str:
    """Build a bank-specific extraction prompt with typed skeleton."""
    bank_desc = _BANK_DESCRIPTIONS.get(bank, bank)
    existing_str = ", ".join(existing_anchors) if existing_anchors else "(none)"
    anchor_list = ", ".join(valid_anchors)

    meta_parts = []
    for k in ("author", "pipeline_tag", "param_count", "family", "vibe_summary"):
        v = existing_metadata.get(k)
        if v:
            meta_parts.append(f"{k}: {v}")
    meta_str = "\n".join(meta_parts) if meta_parts else "(none)"

    prompt = f"""/no_think
You are extracting {bank} information about ML model "{model_id}" from web sources.

Existing knowledge about this model:
{meta_str}

Web evidence (excerpts from web pages about this model):
---
{web_content[:6000]}
---

Your task: select which of these {bank} anchors ({bank_desc}) are SUPPORTED BY the web evidence above.

Valid anchors for {bank}: [{anchor_list}]

Already assigned (do NOT re-select these): [{existing_str}]

RULES:
- ONLY select anchors that have clear evidence in the web text above
- Do NOT guess or infer — if the text doesn't mention it, don't select it
- Select 0-5 NEW anchors (not already assigned)
- For each anchor, quote a brief phrase from the web text as evidence"""

    if bank == "QUALITY":
        prompt += """
- Also extract any benchmark scores mentioned (e.g., MMLU: 73.2, HumanEval: 45.1)"""

    prompt += """

Respond with valid JSON:
{
  "selected_anchors": ["anchor-1", "anchor-2"],
  "evidence": {"anchor-1": "brief quote from text", "anchor-2": "brief quote"},"""

    if bank == "QUALITY":
        prompt += """
  "benchmark_scores": {"benchmark_name": 73.2},"""

    prompt += """
  "summary_supplement": "one sentence of new information not in existing knowledge"
}

If no new anchors are supported by the evidence, return:
{"selected_anchors": [], "evidence": {}, "summary_supplement": ""}"""

    return prompt


def _parse_bank_result(
    text: str, valid_anchors: set[str], existing_anchors: set[str]
) -> dict:
    """Parse and validate one bank extraction result."""
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")

    # Filter anchors to valid set, exclude already-assigned
    raw = data.get("selected_anchors") or []
    if not isinstance(raw, list):
        raw = []
    cleaned = []
    for a in raw:
        if not isinstance(a, str):
            continue
        a = a.strip().lower()
        if a in valid_anchors and a not in existing_anchors and len(a) >= 3:
            cleaned.append(a)

    evidence = data.get("evidence") or {}
    if not isinstance(evidence, dict):
        evidence = {}
    # Keep only evidence for accepted anchors
    evidence = {k: str(v)[:200] for k, v in evidence.items() if k in set(cleaned)}

    benchmarks = data.get("benchmark_scores") or {}
    if not isinstance(benchmarks, dict):
        benchmarks = {}
    # Validate benchmark values are numeric
    valid_benchmarks = {}
    for name, val in benchmarks.items():
        if isinstance(name, str) and isinstance(val, (int, float)):
            if 0 <= val <= 100:
                valid_benchmarks[name.lower().strip()] = round(float(val), 2)

    supplement = data.get("summary_supplement") or ""
    if not isinstance(supplement, str):
        supplement = ""

    return {
        "selected_anchors": cleaned[:5],
        "evidence": evidence,
        "benchmark_scores": valid_benchmarks,
        "summary_supplement": supplement.strip()[:300],
    }


def _ollama_chat(
    base_url: str,
    model_name: str,
    prompt: str,
    temperature: float = 0.3,
    num_predict: int = 512,
    num_ctx: int = 8192,
    timeout: int = 60,
) -> str:
    """Call Ollama native /api/chat with think=false and JSON format.

    Uses the native API instead of OpenAI-compat because the OpenAI layer
    doesn't pass through think=false, causing qwen3.5 to waste all tokens
    on chain-of-thought reasoning instead of producing output.
    """
    payload = json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "format": "json",
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    data = json.loads(resp.read())
    return data.get("message", {}).get("content", "")


def extract_bank(
    ollama_url: str,
    model_name: str,
    model_id: str,
    bank: str,
    valid_anchors: list[str],
    existing_anchors: list[str],
    existing_metadata: dict,
    web_content: str,
    temperature: float = 0.3,
) -> dict | None:
    """Run LLM extraction for one bank. Returns parsed result or None on error."""
    prompt = _build_extraction_prompt(
        model_id, bank, valid_anchors, existing_anchors, existing_metadata, web_content
    )
    try:
        text = _ollama_chat(ollama_url, model_name, prompt, temperature=temperature)
        # Strip markdown fences if present
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        valid_set = set(a.lower() for a in valid_anchors)
        existing_set = set(a.lower() for a in existing_anchors)
        return _parse_bank_result(text, valid_set, existing_set)
    except Exception as e:
        print(f"  Bank {bank} extraction error: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def _load_skip_set(output_path: str) -> set[str]:
    """Load model_ids from existing output for resume."""
    skip: set[str] = set()
    try:
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    mid = item.get("model_id")
                    if mid:
                        skip.add(mid)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return skip


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_one_model(
    ollama_url: str,
    model_name: str,
    item: dict,
    banks_filter: set[str] | None,
    delay: float,
    max_pages: int,
    timeout: int,
    context_limit: int,
    searxng_url: str | None = None,
    snippets_only: bool = False,
) -> dict:
    """Full pipeline for one model: search → fetch → extract per bank."""
    model_id = item["model_id"]
    search_queries = item.get("search_queries") or []
    existing_metadata = item.get("existing_metadata") or {}
    banks_dict = item.get("banks") or {}
    current_anchors = set(existing_metadata.get("current_anchors") or [])

    # Stage 1+2: Search (and optionally fetch pages)
    all_urls: list[str] = []
    all_pages: list[str] = []
    search_failures = 0

    for query in search_queries:
        if _shutdown:
            break
        results = web_search(query, searxng_url, max_results=max_pages, timeout=timeout)
        if not results:
            search_failures += 1
            if search_failures >= 2:
                backoff = min(delay * (2 ** search_failures), 30.0)
                print(f"  Search backoff: {backoff:.0f}s after {search_failures} failures", file=sys.stderr)
                time.sleep(backoff)
        else:
            search_failures = 0

        for r in results:
            url = r["url"]
            if url in set(all_urls):
                continue
            all_urls.append(url)

            if snippets_only:
                # Use search snippet directly — no page fetch, no 403s
                snippet = r.get("snippet", "")
                title = r.get("title", "")
                if snippet or title:
                    all_pages.append(f"{title}\n{snippet}")
            else:
                page = fetch_page(url, timeout=timeout)
                if page and len(page) > 100:
                    all_pages.append(page)
                time.sleep(delay * 0.5)

        time.sleep(delay)

    # Stage 3: Aggregate content
    web_content = aggregate_content(all_pages, model_id, max_chars=context_limit)

    if not web_content:
        return {
            "model_id": model_id,
            "source_urls": all_urls[:10],
            "web_pages_fetched": len(all_pages),
            "banks": {},
            "web_summary": "",
        }

    # Stage 4: Bank-specific extraction
    bank_results: dict[str, dict] = {}
    supplements: list[str] = []

    for bank, anchors in banks_dict.items():
        if _shutdown:
            break
        if banks_filter and bank not in banks_filter:
            continue
        if not anchors:
            continue

        existing_for_bank = [a for a in current_anchors if a in set(a2.lower() for a2 in anchors)]
        result = extract_bank(
            ollama_url, model_name, model_id, bank, anchors,
            existing_for_bank, existing_metadata, web_content,
        )
        if result:
            bank_results[bank] = result
            if result.get("summary_supplement"):
                supplements.append(result["summary_supplement"])

    # Combine web summary
    web_summary = " ".join(supplements[:3]) if supplements else ""

    return {
        "model_id": model_id,
        "source_urls": all_urls[:10],
        "web_pages_fetched": len(all_pages),
        "banks": bank_results,
        "web_summary": web_summary[:500],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Phase E web enrichment worker")
    parser.add_argument("--input", required=True, help="Input shard JSONL file")
    parser.add_argument("--output", required=True, help="Output results JSONL file")
    parser.add_argument("--model", default=None, help="Ollama model name (default: qwen3.5:4b)")
    parser.add_argument(
        "--url", default=None,
        help="Ollama API base URL (env: OLLAMA_BASE_URL, default: http://localhost:11434)",
    )
    parser.add_argument("--resume", action="store_true", help="Skip already-processed models")
    parser.add_argument(
        "--banks", default=None,
        help="Comma-separated banks to process (default: all in input)",
    )
    parser.add_argument(
        "--searxng", default=None,
        help="SearXNG base URL (env: SEARXNG_URL, e.g. http://localhost:8888)",
    )
    parser.add_argument(
        "--snippets-only", action="store_true",
        help="Use only search snippets (no page fetching — much faster)",
    )
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between web requests")
    parser.add_argument("--max-pages", type=int, default=3, help="Max pages per search query")
    parser.add_argument("--timeout", type=int, default=15, help="HTTP request timeout seconds")
    parser.add_argument(
        "--context-limit", type=int, default=8000,
        help="Max chars of web content for LLM context",
    )
    args = parser.parse_args()

    # Resolve defaults from env
    model_name = args.model or os.environ.get("OLLAMA_MODEL", "qwen3.5:4b")
    # Native Ollama API URL (NOT /v1 — we use /api/chat directly)
    base_url = args.url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    base_url = base_url.rstrip("/").removesuffix("/v1")  # normalize
    searxng_url = args.searxng or os.environ.get("SEARXNG_URL")
    if searxng_url:
        searxng_url = searxng_url.rstrip("/")
        print(f"Using SearXNG: {searxng_url}", file=sys.stderr)
    else:
        print("No SearXNG configured, using DDG (may be rate-limited)", file=sys.stderr)
    banks_filter = set(args.banks.upper().split(",")) if args.banks else None

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # Resume support
    skip_set: set[str] = set()
    if args.resume:
        skip_set = _load_skip_set(args.output)
        print(f"Resume: {len(skip_set)} models already processed", file=sys.stderr)

    count = 0
    errors = 0
    skipped = 0
    total_anchors = 0

    file_mode = "a" if args.resume else "w"
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
            if not model_id:
                continue

            if model_id in skip_set:
                skipped += 1
                continue

            print(f"  Processing: {model_id}", file=sys.stderr)
            try:
                result = process_one_model(
                    base_url, model_name, item, banks_filter,
                    args.delay, args.max_pages, args.timeout, args.context_limit,
                    searxng_url=searxng_url,
                    snippets_only=args.snippets_only,
                )
                # Count anchors found
                for bank_res in result.get("banks", {}).values():
                    total_anchors += len(bank_res.get("selected_anchors", []))
                out = json.dumps(result)
            except Exception as e:
                out = json.dumps({"model_id": model_id, "error": str(e)})
                errors += 1

            fout.write(out + "\n")
            fout.flush()
            count += 1

            if count % 5 == 0:
                print(
                    f"Progress: {count} processed ({errors} errors, "
                    f"{total_anchors} anchors found, {skipped} skipped)",
                    file=sys.stderr,
                )

    print(
        f"Done: {count} processed, {errors} errors, "
        f"{total_anchors} anchors found, {skipped} skipped",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
