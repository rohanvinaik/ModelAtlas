"""Vibe-summary gap-fill worker — one of a pair traversing the corpus from opposite ends.

Fills `model_metadata.vibe_summary` for models where structural data isn't
enough to specify the model on its own (per `gating.should_invoke_llm`).
After each successful summary, re-derives EPA into `vibe_e/vibe_p/vibe_a`
in the same transaction so downstream `navigate()` sees a consistent
signal or none at all.

Pair pattern: two workers run concurrently, one from `--direction top`
(highest-PageRank first) and one from `--direction bottom` (lowest of the
gated candidate set). Both claim from a shared `vibe_gapfill_claims`
table via atomic INSERT OR IGNORE — they meet in the middle without
coordination overhead.

Gate discipline: v0.3.0's `should_invoke_llm` decides which models
actually need the LLM. Models with 6+ well-covered banks are skipped
even when their vibe_summary is missing — pattern-matched anchors are
sufficient specification and the LLM has no marginal value to add.

Not zero-imports (unlike phase_e_worker.py): runs on the local machine
against the local DB, so borrowing `vibe_axes` / `gating` / `db` from
the same venv is honest — the zero-import constraint existed only for
`scp`-to-remote workers.
"""

from __future__ import annotations

import argparse
import json
import signal
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Same-venv imports — see module docstring
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_atlas import db, gating, vibe_axes  # noqa: E402


_shutdown = False


def _sigterm(_signum: int, _frame) -> None:  # noqa: ANN001
    global _shutdown
    _shutdown = True


signal.signal(signal.SIGINT, _sigterm)
signal.signal(signal.SIGTERM, _sigterm)


# ── Claim table ────────────────────────────────────────────────────

_CLAIM_SCHEMA = """
CREATE TABLE IF NOT EXISTS vibe_gapfill_claims (
    model_id TEXT PRIMARY KEY,
    worker TEXT NOT NULL,
    claimed_at TEXT NOT NULL,
    outcome TEXT DEFAULT 'pending',
    error TEXT
);
CREATE INDEX IF NOT EXISTS ix_vibe_gapfill_claims_outcome ON vibe_gapfill_claims(outcome);
"""


def ensure_claim_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_CLAIM_SCHEMA)
    conn.commit()


def build_candidate_pool(
    conn: sqlite3.Connection, top_n: int = 5000
) -> list[tuple[str, float]]:
    """Return `[(model_id, pagerank), ...]` for the gated candidate set.

    Two filters compose: (1) in the top-N by PageRank — the tail is
    noise, filling it doesn't move query rankings; (2) `should_invoke_llm`
    — skip models whose existing structure already specifies them.
    Ordered by PageRank descending so `--direction top` starts at the head.
    """
    rows = conn.execute(
        """SELECT model_id, CAST(value AS REAL) as pr
             FROM model_metadata WHERE key='pagerank'
             ORDER BY pr DESC LIMIT ?""",
        (top_n,),
    ).fetchall()
    if not rows:
        return []
    top_pr_ids = [r["model_id"] for r in rows]
    pr_map = {r["model_id"]: float(r["pr"]) for r in rows}
    ph = ",".join("?" for _ in top_pr_ids)
    have_vibe = {
        r[0]
        for r in conn.execute(
            f"""SELECT model_id FROM model_metadata
                 WHERE key='vibe_summary' AND value IS NOT NULL AND value != ''
                   AND model_id IN ({ph})""",
            top_pr_ids,
        ).fetchall()
    }
    have_epa = {
        r[0]
        for r in conn.execute(
            f"SELECT model_id FROM model_metadata WHERE key='vibe_e' AND model_id IN ({ph})",
            top_pr_ids,
        ).fetchall()
    }
    needs = [m for m in top_pr_ids if m not in have_vibe or m not in have_epa]
    # Gate expensively-but-cheaply: single indexed GROUP BY per model
    gated = [(m, pr_map[m]) for m in needs if gating.should_invoke_llm(conn, m)]
    return gated


def claim_next(
    conn: sqlite3.Connection, pool: list[tuple[str, float]], worker: str, from_end: bool
) -> str | None:
    """Atomically claim the next unclaimed model from the pool.

    `from_end=False` → head-first (highest PR); `True` → tail-first (lowest
    PR). Scan the pool linearly for the first entry not yet in `claims`;
    INSERT OR IGNORE and commit. Returns the model_id or None when the
    pool is exhausted.

    SQLite's `INSERT OR IGNORE` is a natural mutex — two workers can race
    to claim the same row and exactly one wins. The loser scans on and
    tries the next candidate.
    """
    it = reversed(pool) if from_end else iter(pool)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for model_id, _pr in it:
        cur = conn.execute(
            "INSERT OR IGNORE INTO vibe_gapfill_claims (model_id, worker, claimed_at) VALUES (?, ?, ?)",
            (model_id, worker, now),
        )
        if cur.rowcount:
            conn.commit()
            return model_id
    return None


# ── Prompt building ───────────────────────────────────────────────

_PROMPT_TEMPLATE = """You are describing a machine-learning model in one sentence.

Model: {model_id}
Author: {author}
Pipeline: {pipeline}
Tags: {tags}
Parameter count: {param_count}
Existing anchors: {anchors}
PageRank rank: #{pr_rank} of {pr_total} (importance in the model network)
Lineage parents: {parents}

Task: write ONE sentence (max 25 words) capturing this model's distinctive
character — WHAT it is, WHO uses it, and its DISTINCTIVE feel (fast, heavy,
specialized, experimental, etc.). Use plain descriptive language, not marketing.

Return ONLY JSON: {{"vibe_summary": "your sentence here"}}
"""


def build_prompt(conn: sqlite3.Connection, model_id: str, pr_rank: int, pr_total: int) -> str:
    """Build the vibe prompt from the graph context the DB already has.

    All fields are derived from stored data — the LLM summarizes, doesn't
    invent. Anchor labels, lineage parents, tags all come from tables we
    populated in earlier passes; the LLM's only job is to compress that
    into one sentence in natural language.
    """
    model_row = conn.execute(
        "SELECT author FROM models WHERE model_id = ?", (model_id,)
    ).fetchone()
    author = (model_row["author"] if model_row else "") or "unknown"
    # pipeline_tag / library_name / parameter_count_b live in model_metadata
    meta_rows = conn.execute(
        """SELECT key, value FROM model_metadata WHERE model_id = ?
             AND key IN ('pipeline_tag','library_name','parameter_count_b','model_type','supported_languages')""",
        (model_id,),
    ).fetchall()
    meta = {r["key"]: r["value"] for r in meta_rows}
    pipeline = meta.get("pipeline_tag") or "unknown"
    param_count = meta.get("parameter_count_b") or "unknown"
    if param_count != "unknown":
        param_count = f"{param_count}B"
    tags_bits = []
    if meta.get("model_type"):
        tags_bits.append(meta["model_type"])
    if meta.get("library_name"):
        tags_bits.append(meta["library_name"])
    if meta.get("supported_languages"):
        tags_bits.append(f"langs={meta['supported_languages']}")
    tags = ", ".join(tags_bits) or ""
    anchor_rows = conn.execute(
        """SELECT a.label FROM model_anchors ma
             JOIN anchors a ON a.anchor_id = ma.anchor_id
            WHERE ma.model_id = ?
            ORDER BY ma.confidence DESC
            LIMIT 12""",
        (model_id,),
    ).fetchall()
    anchors = ", ".join(r["label"] for r in anchor_rows) or "none"
    parent_rows = conn.execute(
        """SELECT target_id FROM model_links
            WHERE source_id = ? AND relation IN ('fine_tuned_from','quantized_from','merged_from')
            LIMIT 3""",
        (model_id,),
    ).fetchall()
    parents = ", ".join(r["target_id"] for r in parent_rows) or "none (base model)"
    tag_str = ", ".join((tags.split(",") if tags else [])[:10]) or "none"
    return _PROMPT_TEMPLATE.format(
        model_id=model_id,
        author=author,
        pipeline=pipeline,
        tags=tag_str,
        param_count=param_count,
        anchors=anchors,
        pr_rank=pr_rank,
        pr_total=pr_total,
        parents=parents,
    )


# ── Ollama call (grammar-constrained JSON) ────────────────────────

_JSON_SCHEMA = {
    "type": "object",
    "properties": {"vibe_summary": {"type": "string"}},
    "required": ["vibe_summary"],
}


def ollama_generate(
    url: str, model: str, prompt: str, timeout: float = 60.0
) -> str | None:
    """Grammar-constrained one-shot call. Returns the summary string or None.

    Ollama's `format` field takes a JSON schema; the model output is
    guaranteed to parse and match the schema — no post-hoc regex parsing.
    Uses `/api/generate` with `stream=false` so we get one response.
    """
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,  # Qwen3.5's thinking mode eats num_predict — off for fast structured output
            "format": _JSON_SCHEMA,
            "options": {"temperature": 0.3, "num_predict": 200},
        }
    ).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"ollama call failed: {exc}") from exc
    data = json.loads(raw)
    try:
        parsed = json.loads(data.get("response", "{}"))
        return str(parsed.get("vibe_summary", "")).strip() or None
    except json.JSONDecodeError:
        return None


# ── Worker loop ────────────────────────────────────────────────────


def process_one(
    conn: sqlite3.Connection,
    model_id: str,
    pr_rank: int,
    pr_total: int,
    ollama_url: str,
    ollama_model: str,
) -> tuple[str, str | None]:
    """`(outcome, error)`. Outcome ∈ {ok, empty, http_error, json_error, ollama_error}."""
    try:
        prompt = build_prompt(conn, model_id, pr_rank, pr_total)
        summary = ollama_generate(ollama_url, ollama_model, prompt)
    except RuntimeError as exc:
        return ("ollama_error", str(exc)[:200])
    if not summary:
        return ("empty", None)
    if len(summary) < 5 or len(summary) > 500:
        return ("empty", f"length={len(summary)}")
    db.set_metadata(conn, model_id, "vibe_summary", summary, "str")
    epa = vibe_axes.derive_and_store(conn, model_id)
    conn.commit()
    return ("ok", f"epa={epa}" if epa else "no_epa_hits")


def mark_outcome(conn: sqlite3.Connection, model_id: str, outcome: str, err: str | None) -> None:
    conn.execute(
        "UPDATE vibe_gapfill_claims SET outcome = ?, error = ? WHERE model_id = ?",
        (outcome, err, model_id),
    )
    conn.commit()


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--worker", required=True, help="Worker name — appears in claim rows for audit.")
    ap.add_argument(
        "--direction",
        choices=("top", "bottom"),
        required=True,
        help="'top' = start at highest PageRank; 'bottom' = start at lowest of the gated set.",
    )
    ap.add_argument("--url", default="http://localhost:11435")
    ap.add_argument("--model", default="qwen3.5:4b")
    ap.add_argument("--top-n", type=int, default=5000)
    ap.add_argument("--max-iterations", type=int, default=None, help="Stop after N models (debug).")
    ap.add_argument(
        "--report-every", type=int, default=25, help="Emit a progress line every N processed models."
    )
    args = ap.parse_args()

    conn = db.get_connection()
    conn.execute("PRAGMA busy_timeout = 5000")
    ensure_claim_schema(conn)

    pool = build_candidate_pool(conn, top_n=args.top_n)
    pool_total = len(pool)
    print(
        f"[{args.worker}] pool={pool_total} direction={args.direction} model={args.model}",
        flush=True,
    )
    if not pool:
        return 0

    processed = 0
    ok = 0
    start = time.time()
    from_end = args.direction == "bottom"

    while not _shutdown:
        model_id = claim_next(conn, pool, worker=args.worker, from_end=from_end)
        if model_id is None:
            print(f"[{args.worker}] pool exhausted", flush=True)
            break
        # Where does this model sit in the pool ranking? (For prompt context.)
        pr_rank = next(
            (i + 1 for i, (m, _) in enumerate(pool) if m == model_id), 0
        )
        outcome, err = process_one(conn, model_id, pr_rank, pool_total, args.url, args.model)
        mark_outcome(conn, model_id, outcome, err)
        processed += 1
        if outcome == "ok":
            ok += 1
        if processed % args.report_every == 0:
            elapsed = time.time() - start
            rate = processed / elapsed if elapsed else 0.0
            print(
                f"[{args.worker}] processed={processed} ok={ok} "
                f"last={model_id[:50]} outcome={outcome} rate={rate:.2f}/s",
                flush=True,
            )
        if args.max_iterations and processed >= args.max_iterations:
            print(f"[{args.worker}] max_iterations hit", flush=True)
            break

    elapsed = time.time() - start
    print(
        f"[{args.worker}] done. processed={processed} ok={ok} elapsed={elapsed:.0f}s",
        flush=True,
    )
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
