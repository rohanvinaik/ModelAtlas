# ModelAtlas

**Google for open-source AI models. Search by what you *mean*, not by keywords.**

[![CI](https://github.com/rohanvinaik/ModelAtlas/actions/workflows/ci.yml/badge.svg)](https://github.com/rohanvinaik/ModelAtlas/actions/workflows/ci.yml)
[![Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_ModelAtlas&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_ModelAtlas)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_ModelAtlas&metric=coverage)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_ModelAtlas)
[![Tests](https://raw.githubusercontent.com/rohanvinaik/ModelAtlas/badges/.github/badges/test-count.svg)](https://github.com/rohanvinaik/ModelAtlas/actions/workflows/spec-badges.yml)
[![Mutation Kill Rate](https://raw.githubusercontent.com/rohanvinaik/ModelAtlas/badges/.github/badges/mutation-kill-rate.svg)](https://github.com/rohanvinaik/ModelAtlas/actions/workflows/spec-badges.yml)

`~51K models · 192 semantic anchors · certifier-enforced · <100ms queries · no embeddings · no GPU`

You want a small code model with tool-calling. HuggingFace gives you the biggest, most popular code models:

```
Qwen2.5-Coder-32B-Instruct          32B   2,081 likes
Qwen3-Coder-480B-A35B-Instruct     480B   1,353 likes
```

480B parameters is not small. HuggingFace sorts by popularity, and it cannot express "small" as a *direction*. ModelAtlas can:

```python
navigate_models(efficiency=-1, capability=+1, quality=+1,
                require_anchors=["code-generation"],
                prefer_anchors=["tool-calling", "high-downloads"])
```

```
jgebbeken/gemma-4-coder-gguf              3B  | 3B-class, C++-code, GGUF-available    1.829
Qwen/Qwen2.5-1.5B-Instruct              1.5B  | 3B-class, Qwen-family, base-model     1.813
deadbydawn101/gemma-4-E4B-Agentic-...     3B  | 3B-class, Apple-Silicon, Gemma-family 1.808
```

Every result is small, code-focused, and popular. One tool call, ~500 tokens, under 100 ms. `efficiency=-1` is not a word to match. It is a direction in a coordinate system.

---

## Four queries, both systems

HuggingFace is free and works. So the only case worth making is output you can compare, and every block below is a real HF API call and a real `navigate_models` call on the same intent. Nothing is illustrative.

### The floor: ask for a task, get models that do the task

*"A model that captions images."*

```
HF  pipeline_tag=image-to-text&sort=likes    MA  require=[image-understanding]
────────────────────────────────────────     ────────────────────────────────────────
1477  Salesforce/blip-image-captioning-large  2.037  Qwen/Qwen3.6-27B
 931  nlpconnect/vit-gpt2-image-captioning    1.957  openai/clip-vit-base-patch32
 865  Salesforce/blip-image-captioning-base   1.975  Qwen/Qwen2.5-VL-7B-Instruct
 500  microsoft/trocr-base-handwritten        1.972  google/vit-base-patch16-224-in21k
 477  numind/NuMarkdown-8B-Thinking           1.986  Qwen/Qwen3.6-35B-A3B
```

Both sides return real image models. That is the floor, and it is the least interesting thing here — but without it, nothing else counts.

Now read *which* models. HF's top three are BLIP and ViT-GPT2, 2021–2022 vintage. They lead because likes accumulate and never decay, so the ranking measures how long a model has been popular, not whether you should use it. ModelAtlas returns current VLMs. Qwen2.5-VL captions better than BLIP and has for a while. Nobody is going to un-like BLIP.

### Direction: "small" is a coordinate, not a substring

*"A small code model."*

```
HF  search="small code model"&sort=likes
────────────────────────────────────────────────────────────────
   0  G-WOO/model_150mil-CodeBERTa-small-v1
   0  penguinman73/codeparrot-model-small
   0  Shawn156/models-small-codeparrot
   0  codecfakev2/model_LA_WCE_4_14_1e-06_wavtokenizer_small_320_24k
   0  BernardJoshua/codet5-small-text-to-sql-prompt-final_model
```

Zero likes, every one. Strangers' abandoned checkpoints — and a wavtokenizer, an *audio* model, because "small" appeared in the filename.

```
MA  efficiency=-1, require=[code-generation], prefer=[tool-calling, high-downloads]
────────────────────────────────────────────────────────────────
1.853  google/gemma-2-2b-it
1.823  DavidAU/gemma-4-E4B-it-The-DECKARD-Expresso-Universe
1.830  Abiray/gemma-4-E4B-it-heretic-GGUF
1.839  bartowski/gemma-2-2b-it-GGUF
1.825  google/gemma-2-2b-jpn-it
```

`efficiency=-1` is a direction in a coordinate system. It does not know what the word "small" looks like, and it does not need to.

### Intent: you asked for code

*"A code model I can run locally, in GGUF."* Skip keywords and use HF's own structured filters — the strongest form of the query it supports:

```
HF  filter=gguf&pipeline_tag=text-generation&sort=likes
────────────────────────────────────────────────────────────────
3375  google/gemma-7b        ← not a code model
2713  yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF
1250  google/gemma-7b-it     ← not a code model
1208  google/gemma-2b        ← not a code model
 933  google/gemma-2b-it     ← not a code model
```

The filter was honoured. The question was ignored. Four of the top five are general-purpose Gemma, because `sort=likes` cannot know you meant *code*. It only knows what is popular among things that survived the filter.

### Choice: ten results, ten models

Same intent, by keyword:

```
HF  search="code gguf"&sort=likes
────────────────────────────────────────────────────────────────
2713  yuxinlu1/gemma-4-12B-coder-...-GGUF
 804  unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF
 767  unsloth/Qwen3-Coder-Next-GGUF
 595  DavidAU/Qwen3.6-40B-...-NEO-CODE-Di-IMatrix-MAX-GGUF
 398  DavidAU/Qwen3.6-27B-...-NEO-CODE-Di-IMatrix-MAX-GGUF
 396  DavidAU/GLM-4.7-Flash-...-NEO-CODE-Imatrix-MAX-GGUF
 329  Jackrong/Qwopus3.6-27B-Coder-MTP-GGUF
 325  Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
 309  Jackrong/Qwopus3.5-9B-Coder-GGUF
 266  Qwen/Qwen3-Coder-Next-GGUF
```

Ten results, five publishers. DavidAU ×3, unsloth ×2, Jackrong ×2, Qwen ×2 — one person's merge recipe, three times.

```
MA  require=[code-generation, GGUF-available], quality=+1
────────────────────────────────────────────────────────────────
1.339  huihui-ai/Huihui-gpt-oss-20b-BF16-abliterated
1.261  sweepai/sweep-next-edit-1.5B              ← predicts your next edit
1.317  defog/sqlcoder-7b-2                       ← writes SQL
1.287  bartowski/Codestral-22B-v0.1-GGUF
1.310  yuxinlu1/gemma-4-12B-coder-...-GGUF
1.290  stabilityai/stable-code-3b
1.261  TheBloke/phi-2-GGUF
1.258  prism-ml/Bonsai-4B-gguf
1.259  mradermacher/gemma-4-19b-a4b-it-REAP-i1-GGUF
1.289  FINAL-Bench/Darwin-28B-Coder-GGUF
```

Ten results, ten publishers, 1.5B to 28B. Two are specialists no keyword reaches: one writes SQL, one predicts your next edit. The scores are not in descending order, and that is the diversification working, not a bug. Score decides who makes the window; MMR decides the order within it, picking greedily on `relevance × (λ − (1−λ)·similarity-to-everything-already-picked)`. So `sweep-next-edit-1.5B` at 1.261 is promoted above `sqlcoder-7b-2` at 1.317 for being *unlike* the pick above it. The list spends its ten slots on ten models, not ten copies.

### Honesty: it says when it can't rank

```
MA  require=[code-generation, GGUF-available], prefer=[tool-calling, high-downloads]
────────────────────────────────────────────────────────────────
1.938  ─      huihui-ai/Huihui-gpt-oss-20b-BF16-abliterated
1.824  tie#0  ubergarm/Kimi-K2.6-GGUF                       axis=EFFICIENCY
1.830  tie#0  TheBloke/Mistral-7B-Instruct-v0.2-GGUF        axis=EFFICIENCY
1.827  tie#0  bartowski/gemma-2-2b-it-GGUF                  axis=EFFICIENCY
1.805  tie#0  DavidAU/Qwen3.6-40B-...-NEO-CODE-MAX-GGUF     axis=EFFICIENCY
```

One clear winner, then a band the engine declines to order. Your constraints do not separate them, so it will not pretend they do. Every ranked list you have ever used presented its arbitrary tail as a ranking. This one names the tie and names the way out:

> *"These range from -2 to +3 on EFFICIENCY, which is rather a wide field. Would you prefer smaller, or larger?"*
> `{"answer": "smaller", "apply": {"efficiency": -1}}`

Answer one word, merge the patch, the band resolves. [How the refinement loop works →](#refining-a-query)

Both systems clear the floor. Past it, every dimension that matters — recency, direction, intent, choice, and knowing what it does not know — needs coordinates, and a keyword index has none.

---

## What the LLM gets

ModelAtlas is an MCP tool. An LLM calls it mid-conversation, and one call returns a full profile:

```json
{
  "model_id": "jgebbeken/gemma-4-coder-gguf",
  "score": 1.8292,
  "score_breakdown": {
    "bank_alignment": 1.0, "anchor_relevance": 1.0, "seed_similarity": 1.0,
    "coherence": 1.0, "pagerank_boost": 1.0007, "soft_combined": 1.8292
  },
  "positions": {
    "CAPABILITY": "+1", "COMPATIBILITY": "+2", "DOMAIN": "+2",
    "EFFICIENCY": "-1", "LINEAGE": "+3", "QUALITY": "+1"
  },
  "anchors": ["3B-class", "C++-code", "GGUF-available", "Gemma-family",
              "base-model", "chat-template-available"]
}
```

From this the LLM immediately knows the model is small, code-focused, GGUF-packaged for local inference, and Gemma-derived. The anchors are a vibe. The positions are a profile. The score explains *why this model and not another*. Without ModelAtlas, the LLM guesses from stale training data. With it, the LLM has live, structured awareness of ~51K models for ~500 tokens — less than the cost of a follow-up question.

| Approach | Latency | Tokens | Quality |
|----------|---------|--------|---------|
| LLM guessing from training data | 0 ms | 0 | Stale, incomplete, no niche coverage |
| HuggingFace API + parse | 2–5 s | ~2,000 | Tag filter + popularity sort |
| **ModelAtlas** | **<100 ms** | **~500** | **Scored, ranked, auditable, certifier-verified** |

### Refining a query

Because the coordinate system is explicit, the engine can tell you which coordinates you did *not* specify. Every response carries a `refine` block naming the highest-value unspecified dimension and the delta that answers it:

```json
"refine": {
  "question": "These range from +0 to +3 on DOMAIN, which is rather a wide
                field. Would you prefer general knowledge, or domain-specialized?",
  "options": [{"answer": "general knowledge", "apply": {"domain": 0}},
              {"answer": "domain-specialized", "apply": {"domain": 1}}],
  "ranking_degraded": false,
  "unspecified_axes": [{"bank": "DOMAIN", "range": "+0..+3", "spread": 1.2}]
}
```

The caller merges `apply` into the arguments it already sent — scalars replace, lists append — and re-calls. No query rebuild. `unspecified_axes` ranks all eight banks by the variance actually present in the result window and drops any bank where every result agrees, so it never asks a question that would not narrow anything. Options come from the observed range, never a fixed ±1: a window at `+0..+3` offers `{"domain": 0}` vs `{"domain": 1}`, because offering `-1` would return an empty set. `ranking_degraded: true` means no `prefer_anchors` were passed, so three of the five soft signals score identically for every candidate — the window is filtered correctly but not meaningfully ordered, and it says so. A keyword engine can offer none of this, because it has no coordinates to be missing.

---

## How it works

Eight signed dimensions. Each has a zero state — the thing most queries assume by default.

```
ARCHITECTURE    zero = transformer decoder       →  +novel (Mamba, MoE)
CAPABILITY      zero = general language model     →  +rich (code, tools, reasoning)
EFFICIENCY      zero = ~7B parameters             →  +larger  / -smaller
COMPATIBILITY   zero = PyTorch + transformers     →  +specific (GGUF, MLX)
LINEAGE         zero = base/foundational model    →  +derived (fine-tune, quant)
DOMAIN          zero = general knowledge           →  +specialized (code, medical)
QUALITY         zero = established mainstream      →  +trending  / -legacy
TRAINING        zero = standard supervised (SFT)  →  +complex (RLHF, DPO) / -simpler
```

On top of coordinates, models share **anchors** — 192 semantic labels like `tool-calling`, `GGUF-available`, `Llama-family`. Similarity is emergent from shared labels, weighted by rarity (IDF). Nothing is an opaque embedding, and every score traces back to specific anchors.

Scoring is multiplicative: `bank_alignment × anchor_relevance × seed_similarity × coherence × context_bias × epa_alignment × soft_combined`. A model that nails efficiency but misses capability gets zero, not fifty percent. Wrong-direction models decay hyperbolically. Avoided anchors stack exponentially — each one halves the score. Required anchors are hard filters. The `coherence` factor is the certifier's per-model verification (below), and `soft_combined` folds the information-theoretic signals (PageRank, PMI-match, IDF-rarity, absence-bonus) together submodularly. Those signals reward rather than filter, so scores are not bounded at 1.0. The result is a surface that favours precise matches and eliminates mismatches fast, with no binary cutoffs. [Full scoring math →](docs/DESIGN.md)

## The audit pipeline: certifier-enforced anchor emissions

One discipline holds the whole corpus together: **every anchor on a model must trace to a structural HuggingFace fact.** Not "the LLM inferred it from the name." Not "the web scrape saw the word nearby." A rule fires on a specific HF field — `pipeline_tag`, `model_type`, `library_name`, quantization level, the safetensors index, `config.json` — and either requires an anchor or forbids one. The certifier lives at `src/model_atlas/certifier/` and enforces at every write path: extract-and-store ingest, Phase E web-enrichment merge, and the retroactive recert tool.

```
    HF raw_json                    Deterministic anchors (config.json → decoder-only, GQA)
        │                               │
        ▼                               ▼
    HFFacts ──────────────────► AnchorEmission[]  (typed, immutable, with Provenance)
                                        │
                                        ▼
                                   certify()      ── 43 declarative Rule objects
                                        │
                     ┌──────────────────┼──────────────────┐
                     ▼                  ▼                  ▼
                 CERTIFIED           REJECTED           AUTO_ADDED
              (write as-is)      (Tier-1 veto)    (rule required it,
                                                    extractor missed it)
```

Rules are declarative data, not code:

```python
Rule(
    name="pipeline_image_text_to_text",
    tier=RuleTier.STRUCTURAL,
    trigger=lambda f: f.pipeline_tag == "image-text-to-text",
    requires=("multimodal", "image-understanding"),
    forbids=("image-generation",),
    reason_template="pipeline_tag=image-text-to-text implies image UNDERSTANDING, not generation",
)
```

43 rules across 8 categories cover the common structural implications, and their tier is their evidence trust. **Tier 1 (STRUCTURAL)** triggers on an HF-published field; a contradiction from a Tier-3 emission (LLM inference, web scrape) is REJECTED, non-negotiable. **Tier 2 (SEMI-STRUCTURAL)** is tag conventions and family names in the repo; a contradiction is DEMOTED. **Tier 3 (INFERRED)** is advisory; a contradiction emits a WARNING and the emission survives. Every model carries a `certification_score`, and 99.95% score ≥ 0.99 — no contradictions surfaced. The rest are surfaced in `navigate_models` as a soft tiebreaker: coherent evidence ranks above internally-contradictory evidence when all other constraints match. Two more guards keep the LLM honest. Before invoking the Phase C/E model, gating checks whether the deterministic tiers already cover ≥ 6 of 8 banks above the confidence floor, and skips the call if so — about 20% of them. And the Phase E worker is grammar-constrained: the Ollama call is schema-restricted to the bank's anchor vocabulary, so the model literally cannot emit an off-vocab or wrong-bank label.

## Extraction — five phases, all certifier-enforced

- **Phases A–B — deterministic** (confidence 1.0 / 0.85). Fetch from HuggingFace; classify from config files, tags, and safetensors metadata. Every anchor routes through the certifier before writing, so a pattern-inferred anchor that contradicts a Tier-1 fact (Falcon-family on a non-Falcon repo) is REJECTED at write time.
- **Phase C — constrained LLM** (confidence 0.5). A local model reads each card and selects from the 192-anchor dictionary. It cannot invent labels — the output schema *is* the vocabulary. Invoked only where deterministic coverage is incomplete.
- **Phase D — audit and heal** (confidence 0.6). Deterministic comparison of C anchors against Tier 1/2 ground truth. Since the certifier now enforces what Phase D once only audited, it is used mostly for coverage gaps.
- **Phase E — web enrichment** (confidence 0.4). Searches the open web for signal HF does not publish — benchmark mentions, comparisons, community impressions — under the same constrained, grammar-restricted selection, routed through the certifier before merging.

All workers are standalone, zero-dependency scripts: `scp` to any machine, `--resume` from any crash, shard across as many machines as you have. Two maintenance tools keep the corpus fresh without a re-extraction: a retroactive recertifier (`scripts/recertify_corpus.py`) walks the whole DB and applies drops for contradictions and adds for missing implied anchors, idempotent and dry-run by default; and a targeted re-pull (`scripts/repull_and_reextract.py`) re-fetches fresh HF metadata for models the certifier flags as noisy, wipes their anchors, and re-extracts through the certified pipeline. [`docs/pipeline.md`](docs/pipeline.md) has the full command reference.

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/rohanvinaik/ModelAtlas.git && cd ModelAtlas && uv sync

# 2. Download the pre-built network (~51K models, all extraction tiers + certifier applied)
mkdir -p ~/.cache/model-atlas
curl -L -o ~/.cache/model-atlas/network.db \
  https://github.com/rohanvinaik/ModelAtlas/releases/latest/download/network.db
```

```json
{
  "mcpServers": {
    "model-atlas": {
      "command": "uv",
      "args": ["--directory", "/path/to/ModelAtlas", "run", "model-atlas"]
    }
  }
}
```

Add that to any MCP-compatible client (Claude Code, Cursor, VS Code). Your LLM can now see model space.

## Tools

| Tool | What it does |
|------|-------------|
| `navigate_models` | **Primary.** Bank directions + anchor constraints + context + EPA target + `mode` + `bank_weights` → scored, ranked results (coherence-weighted, PageRank-boosted, Monty-Hall-sharpened, MMR-diversified, tie-clusters named), plus a [`refine`](#refining-a-query) block. |
| `hf_get_model_detail` | Full profile of one model: all 8 positions, anchors, lineage, `certification_score` |
| `hf_compare_models` | Structural diff between two models: shared/unique anchors, position deltas, Jaccard |
| `hf_search_models` | Natural-language fallback with fuzzy matching when a structured query is not needed |
| `hf_build_index` | Ingest new models from HuggingFace or Ollama (certifier-enforced) |
| `hf_index_status` | Network statistics: model count, anchor distribution, coverage |
| `set_model_vibe` | Set or update a model's vibe summary and extra anchors |
| `phase_e_status` | Web-enrichment progress: enriched count, benchmark scores, recent runs |

## What this is not

- **Not a vector store.** No embeddings. Similarity comes from shared structure.
- **Not a HuggingFace wrapper.** HF is a data source; the value is the extracted structure HF does not expose, plus the certifier that keeps it honest.
- **Not a leaderboard.** There is no "best model" score. Navigation, not ranking.

## Operational discipline

Every write to a canonical table goes through one of two audit-logged primitives in `src/model_atlas/admin.py` — `patch_field` (single-field update, dry-run by default, requires a sourced rationale) and `insert_canonical` (new row, same discipline). Worker JSONL ingestion routes through `reconcile_file()`, which dispatches via the same primitives with SHA-256 line-hash idempotency. Every anchor emission passes the certifier before it hits those primitives, and every successful write appends one line to `data/patches.jsonl`. See [`docs/admin.md`](docs/admin.md), [`docs/reconciler.md`](docs/reconciler.md), and [`docs/coherence.md`](docs/coherence.md).

## Status

Around 51,000 models across 8 signed banks, on a closed vocabulary of 192 anchors and ~406K bank positions. 99.6% bank coverage. 43 declarative rules enforced at every write, 100% of models certifier-scored, 99.95% at perfect coherence. Models with fewer than 5 likes are not indexed — the corpus is the active, community-validated portion of HuggingFace, and it is a periodic snapshot: it tells you *what to look at*, not *what is trending this minute*.

| | |
|---|---|
| Full docs | [rohanv.me/ModelAtlas](https://rohanv.me/ModelAtlas/) |
| Pipeline reference | [`docs/pipeline.md`](docs/pipeline.md) |
| Design deep dive | [`docs/DESIGN.md`](docs/DESIGN.md) |
| Niche query showcase | [`docs/comparison.md`](docs/comparison.md) |

---

MIT — Rohan Vinaik
