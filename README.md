# ModelAtlas

**Google for open-source AI models.** Search by what you *mean*, not by keywords.

[![CI](https://github.com/rohanvinaik/ModelAtlas/actions/workflows/ci.yml/badge.svg)](https://github.com/rohanvinaik/ModelAtlas/actions/workflows/ci.yml)
[![Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_ModelAtlas&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_ModelAtlas)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_ModelAtlas&metric=coverage)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_ModelAtlas)
[![Tests](https://raw.githubusercontent.com/rohanvinaik/ModelAtlas/badges/.github/badges/test-count.svg)](https://github.com/rohanvinaik/ModelAtlas/actions/workflows/spec-badges.yml)
[![Mean σ](https://raw.githubusercontent.com/rohanvinaik/ModelAtlas/badges/.github/badges/sigma.svg)](https://github.com/rohanvinaik/ModelAtlas/actions/workflows/spec-badges.yml)
<br>
[![Mutation Kill Rate](https://raw.githubusercontent.com/rohanvinaik/ModelAtlas/badges/.github/badges/mutation-kill-rate.svg)](https://github.com/rohanvinaik/ModelAtlas/actions/workflows/spec-badges.yml)
[![MC/DC](https://raw.githubusercontent.com/rohanvinaik/ModelAtlas/badges/.github/badges/mcdc.svg)](https://github.com/rohanvinaik/ModelAtlas/actions/workflows/spec-badges.yml)
[![Mutation Sampling](https://raw.githubusercontent.com/rohanvinaik/ModelAtlas/badges/.github/badges/mutation-sampling.svg)](https://github.com/rohanvinaik/ModelAtlas/actions/workflows/spec-badges.yml)

`50,907 models · 192 semantic anchors · 555K anchor links · certifier-enforced · <100ms queries · No embeddings · No GPU`

Fifty thousand models worth looking at live on HuggingFace, and the only ways in are by name, by tag, and by popularity. That holds up right until what you want is a *direction* instead of a word — *small*, *distilled from a frontier model's reasoning*, *local in GGUF*, *multilingual but not code*. A keyword index has nowhere to put a direction, so it hands you a stranger's abandoned checkpoint with the right word in its filename.

ModelAtlas gives every model coordinates instead — and because the coordinates are explicit, it knows what you *didn't* say. Ask for a tiny model that reasons, and when the top candidates tie, it won't fake a rank. It asks:

```
navigate_models(capability=+1, efficiency=-1,
                require_anchors=["distilled", "edge-deployable"],
                prefer_anchors=["reasoning"])

  strong candidates, tied — it won't fake a rank, so it asks:
  › TRAINING is a wide field. Simpler (SFT), or complex (RLHF, DPO)?
    you: simpler
  › QUALITY is a wide field. Established, or trending?
    you: trending

  →  Jackrong/Qwen3.5-0.8B-Claude-4.6-Opus-Reasoning-Distilled-GGUF   1.726
       0.8B · distilled from Claude 4.6 Opus · GGUF · edge-deployable
     every other candidate falls to ~0.87 — they aren't trending
```

Two words — *simpler*, *trending* — and it hands you an 0.8-billion-parameter model that distilled a frontier model's reasoning down to something that runs on your laptop, quantized and ready. You could not have found it by name; its name is a hash of jargon. **No model ran to produce that answer.** ModelAtlas is a deterministic script with no machine learning anywhere near the answer path — a map, not a model — that nonetheless *converses*: it hands you the tied candidates and the one question that separates them, and narrows as you answer. A chatbot from before chatbots could exist, and the conversation *is* the search.

---

## Four queries, both systems

That was one query. The honest case for a tool that sits next to a free one is output you can compare, so here are four, and every block below is a real call to HF's live API beside a real `navigate_models` call on the same intent. Nothing is illustrative.

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

Both sides return real image models. That is the floor, the least interesting thing here — but without it nothing else counts.

Now read *which* models. HF's top three are BLIP and ViT-GPT2, 2021–2022 vintage. They lead because likes accumulate and never decay: the ranking measures **how long a model has been popular**, not whether it is the one to use. ModelAtlas returns current VLMs. Qwen2.5-VL captions better than BLIP and has for a while — but nobody on HF is going to un-like BLIP.

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

`efficiency=-1` is a direction in a coordinate system. It has no idea what the word "small" looks like, and it does not need one.

### Intent: you asked for code

*"A code model I can run locally, in GGUF."* Skip keywords — use HF's own structured filters, the strongest form of the query it supports:

```
HF  filter=gguf&pipeline_tag=text-generation&sort=likes
────────────────────────────────────────────────────────────────
3375  google/gemma-7b        ← not a code model
2713  yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF
1250  google/gemma-7b-it     ← not a code model
1208  google/gemma-2b        ← not a code model
 933  google/gemma-2b-it     ← not a code model
```

The filter was honoured; the question was ignored. Four of the top five are general-purpose Gemma, because `sort=likes` cannot know you meant *code* — only what is popular among things that survived the filter.

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

Ten results, **five publishers**. DavidAU ×3, unsloth ×2, Jackrong ×2, Qwen ×2 — one person's merge recipe, three times.

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

Ten results, **ten publishers**, 1.5B to 28B. Two are specialists no keyword reaches: one writes SQL, one predicts your next edit.

The scores are not in descending order, and that is the diversification working, not a bug. Score decides who makes the window; MMR decides the order within it, picking greedily on `relevance × (λ − (1−λ)·similarity-to-everything-already-picked)`. So `sweep-next-edit-1.5B` at 1.261 is promoted above `sqlcoder-7b-2` at 1.317 for being *unlike* the pick above it. The list spends its ten slots on ten models instead of ten copies of one.

### Honesty: it says when it can't rank

```
MA  require=[code-generation, GGUF-available], prefer=[tool-calling, high-downloads]
────────────────────────────────────────────────────────────────
1.938  ─      huihui-ai/Huihui-gpt-oss-20b-BF16-abliterated
1.824  tie#0  ubergarm/Kimi-K2.6-GGUF                       axis=EFFICIENCY
1.830  tie#0  TheBloke/Mistral-7B-Instruct-v0.2-GGUF        axis=EFFICIENCY
1.827  tie#0  bartowski/gemma-2-2b-it-GGUF                  axis=EFFICIENCY
1.805  tie#0  DavidAU/Qwen3.6-40B-...-NEO-CODE-MAX-GGUF     axis=EFFICIENCY
1.802  tie#0  MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF   axis=EFFICIENCY
1.813  tie#0  unsloth/Kimi-K2.6-GGUF                        axis=EFFICIENCY
1.808  tie#0  Mia-AiLab/Qwable-3.6-27b                      axis=EFFICIENCY
```

One clear winner, then seven the engine **declines to order** — your constraints don't separate them, so it won't pretend they do. Every ranked list you have ever used presented its arbitrary tail as a ranking. This one names the tie and names the way out — the same move as the hero, in the general case:

> *"These range from -2 to +3 on EFFICIENCY, which is rather a wide field. Would you prefer smaller, or larger?"*
> `{"answer": "smaller", "apply": {"efficiency": -1}}`

Answer one word, merge the patch, the band resolves. [How the refinement loop works →](#refining-a-query)

Both systems clear the floor. Past it, every dimension that matters — recency, direction, intent, choice, and knowing what it doesn't know — needs coordinates, and a keyword index has none.

---

## What the LLM gets

This is an MCP tool; an LLM calls it mid-conversation. One call returns, per model:

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

From this the LLM knows, immediately: small, code-focused, GGUF-packaged for local inference, Gemma-derived. The anchors are a vibe; the positions are a profile; the score says *why this model and not another.* Without ModelAtlas the LLM guesses from stale training data. With it, the LLM has live, structured awareness of 50,907 models for ~500 tokens — less than the cost of a follow-up question.

| Approach | Latency | Tokens | What you get |
|----------|---------|--------|--------------|
| LLM guessing from training data | 0ms | 0 | Stale, incomplete, no niche coverage |
| HuggingFace API + parse | 2–5s | ~2,000 | Tag filter, popularity sort |
| **ModelAtlas** | **<100ms** | **~500** | Scored, ranked, tie-aware, certifier-checked |

### Refining a query

The hero's back-and-forth is not a special mode; it is what every response carries. Because the coordinate system is explicit, the engine can see which coordinates you *left unspecified*, and it returns a `refine` block naming the highest-value one and the delta that answers it:

```json
"refine": {
  "question": "These range from -2 to +3 on EFFICIENCY, which is rather a wide
                field. Would you prefer smaller, or larger?",
  "options": [{"answer": "smaller", "apply": {"efficiency": -1}},
              {"answer": "larger",  "apply": {"efficiency": 1}}],
  "ranking_degraded": false,
  "unspecified_axes": [{"bank": "EFFICIENCY", "range": "-2..+3", "spread": 2.56}]
}
```

The caller merges `apply` into the arguments it already sent — scalars replace, lists append — and re-calls. No query rebuild. `unspecified_axes` ranks all eight banks by the variance actually present in the result window and drops any bank where every result already agrees, so it never asks a question that wouldn't narrow anything. The options come from the observed range, never a fixed ±1: a window sitting at `+0..+3` offers `{"domain": 0}` vs `{"domain": 1}`, because offering `-1` would return an empty set. And `ranking_degraded: true` is its way of saying no `prefer_anchors` were passed, so the window is correctly filtered but not meaningfully ordered — the same honesty as `tie_cluster_id` on a result. A keyword engine can offer none of this, because it has no coordinates to be missing.

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

On top of coordinates, models share **anchors** — 192 semantic labels like `tool-calling`, `GGUF-available`, `Llama-family`. Similarity is emergent from shared labels, weighted by rarity (IDF); every score traces back to specific anchors, and nothing is an opaque embedding.

**Scoring** is multiplicative: `bank_alignment × anchor_relevance × seed_similarity × coherence × context_bias × epa_alignment × soft_combined`. A model that nails efficiency but misses capability gets zero, not fifty percent. Wrong-direction models decay hyperbolically; avoided anchors stack exponentially (each halves the score); required anchors are hard filters. The `coherence` factor is the certifier's per-model verification (below). `soft_combined` folds the information-theoretic signals — PageRank, PMI-match, IDF-rarity, absence-bonus — together submodularly; because they reward rather than filter, scores are not bounded at 1.0. The surface strongly favors precise matches and rapidly eliminates mismatches, with no binary cutoffs. `mode` (`auto`/`canonical`/`niche`/`balanced`) reweights the same query — `canonical` surfaces the known incumbents first, `niche` the specialist fits — and `bank_weights` overrides per-bank exponents directly. [Full scoring math →](docs/DESIGN.md) · [v0.4.1 scoring layer →](https://github.com/rohanvinaik/ModelAtlas/releases/tag/v0.4.1)

## The map stays honest: the certifier

Coordinates are only worth trusting if the labels under them are true, so every anchor on every model has to be *earned*. A rule fires on a specific structural HuggingFace fact — `pipeline_tag`, `model_type`, `library_name`, `quantization_level`, the safetensors index, `config.json` — and either requires an anchor or forbids one. Nothing is "the LLM inferred it from the name"; nothing is "the web scrape saw the word nearby." A certifier (`src/model_atlas/certifier/`) enforces this at every write path — Phase A/B ingest, Phase E merge, and the retroactive recert tool.

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

Rules are declarative data, not code — 43 of them across 8 categories (`pipeline_tag`, `model_type`, `library_name`, `quantization`, `safetensors`, tag conventions, family-lineage, code-language):

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

Tiers correspond to how far the evidence can be trusted. **Tier 1 (STRUCTURAL)** is an HF-published field: a contradicting inference from a lower tier is REJECTED, non-negotiable. **Tier 2 (SEMI-STRUCTURAL)** is a tag convention or a family word in the repo name: contradictions are DEMOTED. **Tier 3 (INFERRED)** is advisory: contradictions emit a warning and survive. Two guards keep the LLM honest before it ever runs — gating skips the Phase C/E model on any model the deterministic tiers already cover (≥6 of 8 banks above floor, ~20% of calls), and its JSON output is grammar-restricted to the bank's anchor vocabulary, so it *cannot* emit an off-vocab or wrong-bank label. The result: every model carries a `certification_score`, 99.95% of them ≥ 0.99 with no contradictions, and the rest surface as a soft tiebreaker so coherent evidence outranks internally-contradictory evidence when all else matches. Rule catalogue and tier semantics in [`docs/pipeline.md`](docs/pipeline.md).

## Extraction — five phases, all certifier-enforced

Every phase writes at a confidence tier, and a lower tier cannot overwrite a higher one. Every anchor emission — deterministic, LLM, or web — routes through the certifier before it touches the database.

- **Phases A–B — deterministic** (1.0 / 0.85). Fetch from HuggingFace; classify from config files, tags, and safetensors metadata. A pattern-inferred anchor that contradicts a Tier-1 fact (Falcon-family on a non-Falcon repo) is REJECTED at write time.
- **Phase C — constrained LLM** (0.5). A local model reads each card and selects from the 192-anchor dictionary; it cannot invent labels, because the output schema *is* the vocabulary. Invoked only where deterministic coverage is incomplete.
- **Phase D — audit and heal** (0.6). Deterministic comparison of C anchors against Tier 1/2 ground truth. Since the certifier now enforces the invariants Phase D once only audited, it is used mostly for coverage gaps.
- **Phase E — web enrichment** (0.4). Searches the open web for signal HF doesn't publish — benchmark mentions, comparison articles, community impressions — under the same constrained, grammar-restricted selection, then routes through the certifier before merging.

```bash
python -m model_atlas.ingest_cli --phase ab --min-likes 5        # A/B deterministic
python -m model_atlas.ingest_cli --export-c2 4                   # C: export shards → workers → merge
python -m model_atlas.ingest_cli --merge-c2 results_*.jsonl

# Phase E: one-time self-hosted search (aggregates Google/Bing/DDG, no rate limits)
docker run -d --name searxng -p 8888:8080 \
  -v /path/to/settings.yml:/etc/searxng/settings.yml searxng/searxng
python -m model_atlas.ingest_cli --export-e 4 --export-e-banks CAPABILITY,QUALITY
python scripts/phase_e_worker.py --input shard_0.jsonl --output results_0.jsonl \
    --model qwen3.5:4b --searxng http://localhost:8888 --snippets-only --resume
python -m model_atlas.ingest_cli --merge-e results_*.jsonl
```

Two maintenance tools keep the corpus fresh: `scripts/recertify_corpus.py` walks every model and applies drops for structural contradictions and adds for newly-implied anchors (idempotent, dry-run by default) — this is how new rules onboard without a re-extraction; `scripts/repull_and_reextract.py` re-fetches HF metadata for flagged-noisy models and re-extracts through the certified pipeline. All workers are standalone scripts — `scp` to any machine, `--resume` from any crash, shard across as many machines as you have. [`docs/pipeline.md`](docs/pipeline.md) has the full reference.

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/rohanvinaik/ModelAtlas.git && cd ModelAtlas && uv sync

# 2. Download the pre-built network (50K+ models, all extraction tiers + certifier applied)
mkdir -p ~/.cache/model-atlas
curl -L -o ~/.cache/model-atlas/network.db \
  https://github.com/rohanvinaik/ModelAtlas/releases/latest/download/network.db

# 3. Add to your MCP client config (Claude Code, Cursor, VS Code, etc.)
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

Works with any MCP-compatible client. Your LLM can now see model space.

## Tools

| Tool | What it does |
|------|-------------|
| `navigate_models` | **Primary.** Bank directions + anchor constraints + context anchors + EPA target + `mode` + `bank_weights` → scored, ranked results (coherence-weighted, PageRank-boosted, Monty-Hall-sharpened, MMR-diversified, tie-clusters named), plus a [`refine`](#refining-a-query) block naming the highest-value unspecified dimension. |
| `hf_get_model_detail` | Full profile of one model: all 8 positions, anchors, lineage, metadata, `certification_score` |
| `hf_compare_models` | Structural diff between models: shared/unique anchors, position deltas, Jaccard similarity |
| `hf_search_models` | Natural-language fallback with fuzzy matching when a structured query isn't needed |
| `hf_build_index` | Ingest new models from HuggingFace or Ollama into the network (certifier-enforced) |
| `search_models` | Multi-source search (HuggingFace, Ollama, or all) |
| `hf_index_status` | Network statistics: model count, anchor distribution, coverage |
| `set_model_vibe` | Set or update the vibe summary and optional extra anchors for a model |
| `list_model_sources` | List registered source adapters (HuggingFace, Ollama) and their availability |
| `phase_e_status` | Phase E web-enrichment progress: enriched count, benchmark scores, recent runs |

## What this is not

- **Not a vector store.** No embeddings. Similarity comes from shared structure.
- **Not a HuggingFace wrapper.** HF is a data source; the value is the extracted structure HF doesn't expose, plus the certifier that keeps it honest.
- **Not a ranking system.** No "best model" score. Navigation, not leaderboard — which is why a tie is answered with a question, not a coin-flip.

## Operational discipline

Every write to a canonical table (`models`, `model_positions`, `model_links`, `anchors`) goes through one of two audit-logged primitives in `src/model_atlas/admin.py` — `patch_field` (single-field update, dry-run by default, requires a sourced rationale) and `insert_canonical` (new row, same discipline). Worker-driven JSONL ingestion routes through `model_atlas.reconciler.reconcile_file()`, which dispatches via the same primitives with SHA-256 line-hash idempotency and passes every emission through the certifier first. Every successful write appends one line to `data/patches.jsonl`.

```bash
python -m model_atlas.coherence      # read-only health audit: orthogonality, NULL coverage, orphans
./scripts/sync_and_reconcile.sh      # weekly hub-and-spoke sync: spokes → reconciler → audit → rotate
```

See [`docs/admin.md`](docs/admin.md), [`docs/reconciler.md`](docs/reconciler.md), and [`docs/coherence.md`](docs/coherence.md) for the discipline.

## Status

**50,907 models. 192 anchors. 555K model-anchor links. 406K bank positions across 8 banks (144K off the zero state). 99.6% bank coverage. 4,021 models web-enriched. 43 declarative rules enforced at every write. 100% of models certifier-scored, 99.95% at perfect coherence. 933 tests passing.**

Models with fewer than 5 likes are not indexed yet — the 50K are the active, community-validated portion of HuggingFace. A periodic snapshot: it tells you *what to look at*, not *what is trending this minute*.

| | |
|---|---|
| Full docs | [rohanv.me/ModelAtlas](https://rohanv.me/ModelAtlas/) |
| Pipeline reference | [`docs/pipeline.md`](docs/pipeline.md) |
| Design deep dive | [`docs/DESIGN.md`](docs/DESIGN.md) |
| Write primitives | [`docs/admin.md`](docs/admin.md) |
| Reconciler (worker JSONL → canonical) | [`docs/reconciler.md`](docs/reconciler.md) |
| Coherence audit | [`docs/coherence.md`](docs/coherence.md) |
| Niche query showcase | [`docs/comparison.md`](docs/comparison.md) |

---

MIT — Rohan Vinaik
