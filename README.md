# ModelAtlas

**An atlas of open-source AI models.** Not a search box — a map, and something that reads it with you.

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

Every search engine answers a question you didn't quite ask, ranks the results in an order it can't defend, and leaves you to figure out which of the ten near-identical things is yours.

An atlas does something else. It shows you the region you're standing in, tells you honestly where its own borders are, and points at the road that leads somewhere different.

You ask for a code model you can run locally:

```python
navigate_models(require_anchors=["code-generation", "GGUF-available"],
                prefer_anchors=["tool-calling", "high-downloads"], quality=+1)
```

```
1.938  ─      huihui-ai/Huihui-gpt-oss-20b-BF16-abliterated
1.830  tie#0  TheBloke/Mistral-7B-Instruct-v0.2-GGUF          axis=EFFICIENCY
1.827  tie#0  bartowski/gemma-2-2b-it-GGUF                    axis=EFFICIENCY
1.824  tie#0  ubergarm/Kimi-K2.6-GGUF                         axis=EFFICIENCY
1.813  tie#0  unsloth/Kimi-K2.6-GGUF                          axis=EFFICIENCY
1.808  tie#0  Mia-AiLab/Qwable-3.6-27b                        axis=EFFICIENCY
1.805  tie#0  DavidAU/Qwen3.6-40B-...-NEO-CODE-MAX-GGUF       axis=EFFICIENCY
1.802  tie#0  MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF     axis=EFFICIENCY
```

One clear winner — and then a band of seven that ModelAtlas **refuses to rank**, because nothing in your constraints separates them. It won't invent an order it can't earn. Instead it tells you the road out:

> *"These range from -2 to +3 on EFFICIENCY, which is rather a wide field. Would you prefer smaller, or larger?"*
>
> `{"answer": "smaller", "apply": {"efficiency": -1}}`

You answer one word. The `apply` dict merges into the call you already made — you don't rebuild the query — and the band resolves. The tool stopped answering and started interviewing.

That's the whole idea. **ModelAtlas knows what it doesn't know, and asks.**

---

## The same question, asked of both

One intent: *a code model I can run locally, in GGUF.* Both sides are real calls — HuggingFace's live API, ModelAtlas's `navigate_models`. Reproduce them yourself; the exact queries are inline.

### Popularity is not intent

```
GET /api/models?filter=gguf&pipeline_tag=text-generation&sort=likes
```
```
3375  google/gemma-7b          ← not a code model
2713  yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF
1250  google/gemma-7b-it       ← not a code model
1208  google/gemma-2b          ← not a code model
 933  google/gemma-2b-it       ← not a code model
```

You filtered for code. Four of the top five are general-purpose Gemma. The filter was honoured and the *question* was ignored, because `sort=likes` has no idea what you wanted — it only knows what's popular.

### Ten doors, one room

Try keywords instead:

```
GET /api/models?search=code+gguf&sort=likes
```
```
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

Ten results, **five publishers**: DavidAU ×3, unsloth ×2, Jackrong ×2, Qwen ×2. You didn't get ten options. You got one person's uncensored merge recipe three times, and a popularity contest among Qwen repackagings.

The same intent, as coordinates:

```python
navigate_models(require_anchors=["code-generation", "GGUF-available"], quality=+1)
```
```
1.567  huihui-ai/Huihui-gpt-oss-20b-BF16-abliterated
1.512  sweepai/sweep-next-edit-1.5B                  ← edit-prediction specialist
1.551  defog/sqlcoder-7b-2                           ← SQL specialist
1.531  bartowski/Codestral-22B-v0.1-GGUF
1.547  yuxinlu1/gemma-4-12B-coder-...-GGUF
1.533  stabilityai/stable-code-3b
1.512  TheBloke/phi-2-GGUF
1.510  prism-ml/Bonsai-4B-gguf
1.511  mradermacher/gemma-4-19b-a4b-it-REAP-i1-GGUF
1.532  FINAL-Bench/Darwin-28B-Coder-GGUF
```

**Ten publishers out of ten.** 1.5B to 28B. Every one is actually a code model. Two are specialists no keyword reaches — a model that writes SQL, and a model that predicts your next edit. MMR diversification is doing this: the ranker actively suppresses results that are near-duplicates of ones it already picked, because ten addresses on one street is not a choice.

### What it won't do

ModelAtlas is not magic, and the map has edges it will show you.

Ask HuggingFace for `text to speech` and four of the top eight results are speech-**to-text** — the inverse task — because the models actually named after the task are the amateur ones, and the good TTS models (Kokoro, XTTS, Bark) don't contain the string. ModelAtlas does **not** fix this. Its `speech-domain` anchor has no direction either, so it will hand you `nvidia/parakeet-tdt` (an ASR model) next to real TTS. The vocabulary lacks the distinction, so the atlas is blank there — and says so, rather than dressing up a guess.

**The pattern:** HuggingFace answers *what you typed*. ModelAtlas answers *what you constrained*, tells you what you left unconstrained, and admits where its own map runs out.

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

From this, the LLM *immediately knows*: small (3B, negative on EFFICIENCY), code-focused, GGUF-packaged for local inference, Gemma-derived. The anchors are a vibe. The positions are a profile. The `score_breakdown` explains *why this model and not another* — every factor is inspectable, and `soft_combined` shows exactly how much the information-theoretic signals lifted it above a bare constraint match.

## The refinement loop

Every response also carries a `refine` block. This is the part that makes it an atlas rather than a search box: the pipeline is deterministic, so it knows precisely which dimensions its own answer is silent on — and it would rather ask than bluff.

```json
"refine": {
  "question_id": "unconstrained_axis",
  "question": "These range from -2 to +3 on EFFICIENCY, which is rather a wide
                field. Would you prefer smaller, or larger?",
  "options": [
    {"answer": "smaller", "apply": {"efficiency": -1}},
    {"answer": "larger",  "apply": {"efficiency": 1}}
  ],
  "merge_rule": "Merge `apply` into the arguments you already sent; do not
                 rebuild the query. Scalar keys replace; list keys append.",
  "ranking_degraded": false,
  "unspecified_axes": [
    {"bank": "EFFICIENCY", "range": "-2..+3", "spread": 2.56, "distinct": 3},
    {"bank": "DOMAIN",     "range": "+0..+3", "spread": 1.44, "distinct": 2},
    {"bank": "CAPABILITY", "range": "+0..+2", "spread": 0.64, "distinct": 2}
  ],
  "splitting_anchors": [
    {"anchor": "gguf-quantized", "present_in": 4, "out_of": 8}
  ]
}
```

The loop, for a calling LLM:

1. Call `navigate_models` with whatever the user gave you — however vague.
2. Read `refine.question`. It names the single highest-value thing still unspecified.
3. Ask the user, or answer it yourself if the conversation already settles it.
4. **Merge the chosen option's `apply` into the same arguments you just sent.** Don't rebuild the query. Scalars replace; lists append.
5. Re-call. Repeat until `refine.question` is empty.

Three properties make this trustworthy rather than chatty:

**It never asks a useless question.** `unspecified_axes` ranks all eight banks by the variance actually observed in your result window, and drops any bank where every result agrees — asking about those would be noise. The top entry is, by construction, the most informative thing you could say next.

**It never offers an answer that leads nowhere.** Options are derived from the observed range, not a fixed ±1. A window sitting at `+0..+3` on DOMAIN offers *"general knowledge"* (`{"domain": 0}`) versus *"domain-specialized"* (`{"domain": 1}`) — never `-1`, because no result lives there and answering it would return an empty set.

**It tells you when its ranking is meaningless.** `ranking_degraded: true` means you passed no `prefer_anchors`. Three of the five soft signals score identically for every candidate that clears a `require` filter, so the window is correctly *filtered* but not meaningfully *ordered*. Treat it as a set, not a ranking — and the block says so in as many words rather than letting you quietly trust an arbitrary #1.

Questions are declarative skeletons with typed gaps, the same discipline the certifier's `reason_template` uses — `"These range from <range_low> to <range_high> on <bank>..."`, filled deterministically. Switch on the stable `question_id`; the prose may be reworded, the id won't be.

Without ModelAtlas, the LLM guesses from stale training data. With it, the LLM has live, structured awareness of 50,906 models for ~500 tokens — less than the cost of a follow-up question.

| Approach | Latency | Tokens | Quality |
|----------|---------|--------|---------|
| LLM guessing from training data | 0ms | 0 | Stale, incomplete, no niche coverage |
| HuggingFace API + parse | 2-5s | ~2,000 | Tag filter + popularity sort |
| **ModelAtlas** | **<100ms** | **~500** | **Scored, ranked, auditable, certifier-verified** |

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

**Scoring** is multiplicative across seven factors:

```
score = bank_alignment × anchor_relevance × seed_similarity
        × coherence × context_bias × epa_alignment × soft_combined
```

Multiplicative means a model that nails efficiency but misses capability gets zero, not fifty percent. Wrong-direction models decay hyperbolically. Avoided anchors stack exponentially (each halves the score). Required anchors are hard filters. `coherence` comes from the certifier's per-model verification (below).

The seventh factor, `soft_combined`, is where the information-theoretic layer lives. Five signals — PageRank authority, PMI match, IDF rarity, an absence bonus, and a superadditive PageRank×rarity term — are folded together **submodularly**, so each additional signal adds less than the last and no single one can run away with the ranking. Because these signals reward rather than filter, scores are not bounded at 1.0. Monty Hall sharpening drives directly-opposed models to 0.05, and MMR diversification runs over the returned window so the top results aren't near-duplicates.

Every factor is returned in `score_breakdown`, so any ranking can be taken apart after the fact. [Full scoring math →](docs/DESIGN.md)

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

43 rules across 8 categories cover the common structural implications:

| Category | Rules | Category | Rules |
|---|---|---|---|
| `family_evidence` | 11 | `library_name` | 4 |
| `pipeline_tag` | 10 | `quantization` | 3 |
| `model_type` | 9 | `safetensors` | 1 |
| `tag` | 4 | `code_language` | 1 |

Rule tiers correspond to evidence trust:

- **Tier 1 (STRUCTURAL)** — the trigger is an HF-published field. Contradictions from a Tier-3 emission (LLM inference, web scrape) get REJECTED. Non-negotiable. **39 of the 43 rules.**
- **Tier 2 (SEMI-STRUCTURAL)** — tag conventions, family word in the repo name. Contradictions get DEMOTED (confidence lowered). **The remaining 4.**
- **Tier 3 (INFERRED)** — advisory; contradictions emit a WARNING and the emission survives. The machinery supports this tier, but no rule is currently authored at it — a rule is only worth writing when the trigger is a fact, and Tier 3 by definition isn't.

**Result**: all 50,906 models carry a `certification_score` metadata field. 99.95% score ≥ 0.99 (no contradictions surfaced). The remainder are surfaced in `navigate_models` results as a soft ranking tiebreaker — coherent evidence ranks above internally-contradictory evidence when all other constraints match.

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
| `navigate_models` | **Primary.** Bank directions + anchor constraints + context anchors + EPA target + `mode` + `bank_weights` → scored, ranked results (coherence-weighted, PageRank-boosted, Monty-Hall-sharpened, MMR-diversified, tie-clusters named) **plus a `refine` block** naming the highest-value unspecified dimension and the ready-to-merge patch that answers it. See [the refinement loop](#the-refinement-loop). |
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
