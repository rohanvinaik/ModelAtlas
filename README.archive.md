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

`50,906 models · 192 semantic anchors · 555K anchor links · certifier-enforced · <100ms queries · No embeddings · No GPU`

You want a small code model with tool-calling.

**HuggingFace** gives you the biggest, most popular code models:

```
Qwen2.5-Coder-32B-Instruct          32B   2,081 likes
Qwen3-Coder-480B-A35B-Instruct     480B   1,353 likes
```

480B parameters. Not small. HF sorts by popularity. It can't express "small" as a *direction*.

**ModelAtlas** gives you what you actually asked for:

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

Every result is small, code-focused, and popular. One tool call. ~500 tokens. <100ms.

**Since v0.4.1**, `navigate_models` also accepts `mode` (`auto`/`canonical`/`niche`/`balanced`) and `bank_weights` (per-bank exponent overrides). The same hero query under `mode="canonical"` surfaces the well-known incumbents first; under `mode="niche"` it prioritises specialist fits. Scores are information-theoretic (PMI-match, IDF-rare boost, absence-bonus, Monty Hall opposition sharpening, MMR diversification, all combined submodularly) — see [v0.4.1 release notes](https://github.com/rohanvinaik/ModelAtlas/releases/tag/v0.4.1) for the full scoring layer.

---

## Four queries, both systems

HuggingFace is free and works fine. So the only honest case for this is output you can compare. Every block below is a real call to HF's live API and a real `navigate_models` call, run on the same intent. Nothing is illustrative.

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

Both sides return real image models. That is the floor, and it is the least interesting thing here — but without it nothing else counts.

Now read *which* models. HF's top three are BLIP and ViT-GPT2: 2021–2022 vintage. They lead because likes accumulate and never decay — the ranking measures **how long a model has been popular**, not whether it is the one you should use. ModelAtlas returns current VLMs. Qwen2.5-VL captions better than BLIP and has for a while. Nobody on HF is going to un-like BLIP.

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

Zero likes. Every one. Strangers' abandoned checkpoints, and a wavtokenizer — an *audio* model — because "small" appeared in the filename.

```
MA  efficiency=-1, require=[code-generation], prefer=[tool-calling, high-downloads]
────────────────────────────────────────────────────────────────
1.853  google/gemma-2-2b-it
1.823  DavidAU/gemma-4-E4B-it-The-DECKARD-Expresso-Universe
1.830  Abiray/gemma-4-E4B-it-heretic-GGUF
1.839  bartowski/gemma-2-2b-it-GGUF
1.825  google/gemma-2-2b-jpn-it
```

`efficiency=-1` is a direction in a coordinate system. It has no idea what the word "small" looks like.

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

The filter was honoured. The question was ignored. Four of the top five are general-purpose Gemma, because `sort=likes` cannot know that you meant *code* — it only knows what is popular among things that survived the filter.

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

Ten results, **five publishers**. DavidAU ×3, unsloth ×2, Jackrong ×2, Qwen ×2. One person's merge recipe, three times.

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

Ten results, **ten publishers**. 1.5B to 28B. Two are specialists no keyword reaches: one writes SQL, one predicts your next edit.

The scores are not in descending order, and that is the diversification working, not a bug. Score decides who makes the window; MMR decides the order within it, picking greedily on `relevance × (λ − (1−λ)·similarity-to-everything-already-picked)`. So `sweep-next-edit-1.5B` at 1.261 is promoted above `sqlcoder-7b-2` at 1.317 for being *unlike* the pick above it. The list spends its ten slots on ten models instead of ten copies.

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

One clear winner, then seven the engine **declines to order** — your constraints don't separate them, so it won't pretend. Every ranked list you have ever used presented its arbitrary tail as a ranking. This one names the tie and names the way out:

> *"These range from -2 to +3 on EFFICIENCY, which is rather a wide field. Would you prefer smaller, or larger?"*
> `{"answer": "smaller", "apply": {"efficiency": -1}}`

Answer one word, merge the patch, the band resolves. [How the refinement loop works →](#refining-a-query)

**The pattern:** both systems clear the floor. Past it, every dimension that matters — recency, direction, intent, choice, and knowing what it doesn't know — needs coordinates, and a keyword index doesn't have any.

---

## What the LLM gets

This is an MCP tool. An LLM calls it during conversation. One tool call returns:

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

From this, the LLM *immediately knows*: small, code-focused, GGUF-packaged for local inference, Gemma-derived. The anchors are a vibe. The positions are a profile. The score explains *why this model and not another.*

Without ModelAtlas, the LLM guesses from stale training data. With it, the LLM has live, structured awareness of 50,906 models for ~500 tokens — less than the cost of a follow-up question.

| Approach | Latency | Tokens | Quality |
|----------|---------|--------|---------|
| LLM guessing from training data | 0ms | 0 | Stale, incomplete, no niche coverage |
| HuggingFace API + parse | 2-5s | ~2,000 | Tag filter + popularity sort |
| **ModelAtlas** | **<100ms** | **~500** | **Scored, ranked, auditable, certifier-verified** |

### Refining a query

Structure has a second payoff: because the coordinate system is explicit, the engine can tell you which coordinates you *didn't* specify. Every response carries a `refine` block naming the highest-value unspecified dimension and the delta that answers it:

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

The caller merges `apply` into the arguments it already sent — scalars replace, lists append — and re-calls. No query rebuild. `unspecified_axes` ranks all eight banks by the variance actually present in the result window and drops any bank where every result agrees, so it never asks a question that wouldn't narrow anything. Options come from the observed range, never a fixed ±1: a window at `+0..+3` offers `{"domain": 0}` vs `{"domain": 1}`, because offering `-1` would return an empty set.

`ranking_degraded: true` means no `prefer_anchors` were passed — three of the five soft signals then score identically for every candidate that clears the `require` filter, so the window is correctly filtered but not meaningfully ordered. Same discipline as `tie_cluster_id` on a result: when the constraints don't separate two models, say so rather than fake a rank. A keyword engine can't offer any of this, because it has no coordinates to be missing.

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

On top of coordinates, models share **anchors** — 192 semantic labels like `tool-calling`, `GGUF-available`, `Llama-family`. Similarity is emergent from shared labels, weighted by rarity (IDF). Every score traces back to specific anchors. Nothing is an opaque embedding.

**Scoring:** `bank_alignment × anchor_relevance × seed_similarity × coherence × context_bias × epa_alignment × soft_combined`. Multiplicative — a model that nails efficiency but misses capability gets zero, not fifty percent. Wrong-direction models decay hyperbolically. Avoided anchors stack exponentially (each halves the score). Required anchors are hard filters. The `coherence` factor comes from the certifier's per-model verification (below). `soft_combined` folds the information-theoretic signals (PageRank, PMI-match, IDF-rarity, absence-bonus) together submodularly; because they reward rather than filter, scores are not bounded at 1.0. The result is a scoring surface that strongly favors precise matches and rapidly eliminates mismatches, without binary cutoffs. [Full scoring math →](docs/DESIGN.md)

## The audit pipeline: certifier-enforced anchor emissions

The core discipline: **every anchor assigned to a model must be traceable to a structural HuggingFace fact.** Not "the LLM inferred it from the name." Not "the web scrape saw the word in a nearby paragraph." A rule fires on a specific HF field (pipeline_tag, model_type, library_name, quantization_level, safetensors index, config.json), and either requires an anchor or forbids one.

The certifier lives at `src/model_atlas/certifier/` and enforces at every write path — extract-and-store (Phase A/B ingest), Phase E web-enrichment merge, and the retroactive recert tool.

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

43 rules across 8 categories cover the common structural implications: `pipeline_tag` (10), `model_type` (9), `library_name` (4), `quantization` (3), `safetensors` (1), tag conventions (4), family-lineage evidence (11), and code-language (1).

Rule tiers correspond to evidence trust:

- **Tier 1 (STRUCTURAL)** — the trigger is an HF-published field. Contradictions from a Tier-3 emission (LLM inference, web scrape) get REJECTED. Non-negotiable.
- **Tier 2 (SEMI-STRUCTURAL)** — tag conventions, family word in the repo name. Contradictions get DEMOTED (confidence lowered).
- **Tier 3 (INFERRED)** — advisory. Contradictions emit a WARNING; the emission survives.

**Result**: every model in the corpus carries a `certification_score` metadata field. 99.95% of models score ≥ 0.99 (no contradictions surfaced). The remainder are surfaced in `navigate_models` results as a soft ranking tiebreaker — coherent evidence ranks above internally-contradictory evidence when all other constraints match.

**LLM gating** (`src/model_atlas/gating.py`): before invoking the Phase C/E LLM on a model, check whether the deterministic tiers have already covered ≥ 6 of 8 banks with above-floor confidence. If so, skip the LLM. On the tier-2 candidate set, ~20% of LLM calls are skipped this way.

**Grammar-constrained JSON output** (Phase E worker): the Ollama call is schema-restricted to the bank's anchor vocabulary. The LLM literally cannot emit an off-vocab or wrong-bank label — the enum is enforced by Ollama's structured-output mode. Falls back to loose JSON if the runtime doesn't support the schema.

## Extraction — five phases, all certifier-enforced

**Phases A–B: Deterministic extraction** (confidence 1.0 / 0.85). Fetch from HuggingFace, classify from config files, tags, and safetensors metadata. Deterministic anchors are routed through the certifier before writing — any pattern-inferred anchor that contradicts a Tier-1 fact (Falcon-family on a non-Falcon repo, Rust-code on a non-generative pipeline) gets REJECTED at write time.

```bash
python -m model_atlas.ingest_cli --phase ab --min-likes 5
```

**Phase C: Constrained LLM classification** (confidence 0.5). A local model reads each model card and selects from the 192-anchor dictionary. It cannot invent labels — the output schema is the vocabulary. Only invoked for models where deterministic coverage is incomplete.

```bash
python -m model_atlas.ingest_cli --export-c2 4       # export shards
# scp shard files to worker machines; workers are zero-dependency Python scripts
python -m model_atlas.ingest_cli --merge-c2 results_*.jsonl
```

**Phase D: Audit and heal** (confidence 0.6). Deterministic comparison of C2 anchors against Tier 1/2 ground truth. Mismatches get re-extracted. Since the certifier now enforces the same invariants Phase D historically only *audited*, Phase D is used mostly for coverage gaps rather than corrections.

**Phase E: Web enrichment** (confidence 0.4). Phases A–D work from HuggingFace metadata; Phase E searches the open web for signal HF doesn't publish — benchmark mentions, comparison articles, community impressions. Same constrained selection from the anchor vocabulary, grammar-restricted per bank, then routed through the certifier before merging.

```bash
# One-time: self-hosted search (aggregates Google/Bing/DDG, no rate limits)
docker run -d --name searxng -p 8888:8080 \
  -v /path/to/settings.yml:/etc/searxng/settings.yml searxng/searxng

# Export → run → merge (same pattern as C/D; merge routes through certifier via Phase 7)
python -m model_atlas.ingest_cli --export-e 4 --export-e-banks CAPABILITY,QUALITY
python scripts/phase_e_worker.py --input shard_0.jsonl --output results_0.jsonl \
    --model qwen3.5:4b --searxng http://localhost:8888 --snippets-only --resume
python -m model_atlas.ingest_cli --merge-e results_*.jsonl --merge-e-dry-run
python -m model_atlas.ingest_cli --merge-e results_*.jsonl
```

**Retroactive recertification** (`scripts/recertify_corpus.py`). Walks every model in the DB, runs the certifier, applies drops for structural contradictions and adds for missing implied anchors. Idempotent; dry-run by default. Used to onboard new rules to the existing corpus without a re-extraction.

```bash
python scripts/recertify_corpus.py                   # dry-run — report only
python scripts/recertify_corpus.py --apply           # write the diff, audit-logged
```

**Targeted re-pull** (`scripts/repull_and_reextract.py`). For models the certifier flags as noisy, fetch fresh HF metadata, wipe existing anchors, and re-extract via the certified pipeline. Ensures the source of truth stays fresh.

All workers are standalone scripts — `scp` to any machine, `--resume` from any crash, shard across as many machines as you have. [`docs/pipeline.md`](docs/pipeline.md) has the full reference.

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/rohanvinaik/ModelAtlas.git && cd ModelAtlas && uv sync

# 2. Download pre-built network (50K+ models, all extraction tiers + certifier applied)
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
| `navigate_models` | **Primary.** Bank directions + anchor constraints + context anchors + EPA target + `mode` + `bank_weights` → scored, ranked results (coherence-weighted, PageRank-boosted, Monty-Hall-sharpened, MMR-diversified, tie-clusters named), plus a [`refine`](#refining-a-query) block naming the highest-value unspecified dimension. See [v0.4.1 scoring layer](https://github.com/rohanvinaik/ModelAtlas/releases/tag/v0.4.1). |
| `hf_get_model_detail` | Full profile of one model: all 8 positions, anchors, lineage, metadata, `certification_score` |
| `hf_compare_models` | Structural diff between models: shared/unique anchors, position deltas, Jaccard similarity |
| `hf_search_models` | Natural language fallback with fuzzy matching when structured query isn't needed |
| `hf_build_index` | Ingest new models from HuggingFace or Ollama into the network (certifier-enforced) |
| `search_models` | Multi-source search (HuggingFace, Ollama, or all) |
| `hf_index_status` | Network statistics: model count, anchor distribution, coverage |
| `set_model_vibe` | Set/update the vibe summary and optional extra anchors for a model |
| `list_model_sources` | List registered source adapters (HuggingFace, Ollama) and their availability |
| `phase_e_status` | Phase E web-enrichment progress: enriched count, benchmark scores, recent runs |

## What this is not

- **Not a vector store.** No embeddings. Similarity comes from shared structure.
- **Not a HuggingFace wrapper.** HF is a data source. The value is the extracted structure HF doesn't expose plus the certifier that keeps that structure honest.
- **Not a ranking system.** No "best model" score. Navigation, not leaderboard.

## Operational discipline

Every write to a canonical table (`models`, `model_positions`, `model_links`, `anchors`) goes through one of two audit-logged primitives in `src/model_atlas/admin.py`:

- `patch_field(table, pk, field, old, new, reason)` — single-field update, dry-run by default, requires sourced rationale.
- `insert_canonical(table, row, reason)` — new row insert, same discipline.

Worker-driven JSONL ingestion routes through `model_atlas.reconciler.reconcile_file()` which dispatches via the same primitives with SHA-256 line-hash idempotency. Every anchor emission — from deterministic extraction, from Phase E merge, from targeted re-pull — passes through the certifier before hitting these primitives. Every successful write appends one line to `data/patches.jsonl`.

```bash
# Health audit (read-only): bank orthogonality, NULL coverage, anchor orphans/oversaturation
python -m model_atlas.coherence

# Weekly hub-and-spoke sync: rsync from spokes → reconciler → audit → rotate log
./scripts/sync_and_reconcile.sh
```

See [`docs/admin.md`](docs/admin.md), [`docs/reconciler.md`](docs/reconciler.md), and [`docs/coherence.md`](docs/coherence.md) for the discipline.

## Status

**50,906 models. 192 anchors. 555K model-anchor links. 406K bank positions across 8 banks (144K off the zero state). 99.6% bank coverage. 4,021 models web-enriched. 43 declarative rules enforced at every write. 100% of models certifier-scored (99.95% at perfect coherence). 933 tests passing.**

Models with < 5 likes are not yet indexed — the 50K represent the active, community-validated portion of HuggingFace. Periodic snapshot — tells you *what to look at*, not *what's trending right now*.

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
