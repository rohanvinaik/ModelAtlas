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

`50,906 models · 192 semantic anchors · 377K anchor links · certifier-enforced · <100ms queries · No embeddings · No GPU`

You want a small code model with tool-calling.

**HuggingFace** gives you the biggest, most popular code models:

```
Qwen2.5-Coder-32B-Instruct          32B   1,996 likes
Qwen3-Coder-480B-A35B-Instruct     480B   1,315 likes
```

480B parameters. Not small. HF sorts by popularity. It can't express "small" as a *direction*.

**ModelAtlas** gives you what you actually asked for:

```python
navigate_models(efficiency=-1, capability=+1, quality=+1,
                require_anchors=["code-generation"],
                prefer_anchors=["tool-calling", "high-downloads"])
```

```
jgebbeken/gemma-4-coder-gguf              3B  | code, GGUF, function-calling      1.83
Qwen/Qwen2.5-1.5B-Instruct              1.5B  | code, high-downloads, Qwen-family 1.81
deadbydawn101/gemma-4-E4B-...-MLX         3B  | code, MLX-compatible              1.81
```

Every result is small, code-focused, and popular. One tool call. ~500 tokens. <100ms.

**Since v0.4.1**, `navigate_models` also accepts `mode` (`auto`/`canonical`/`niche`/`balanced`) and `bank_weights` (per-bank exponent overrides). The same hero query under `mode="canonical"` surfaces the well-known incumbents first; under `mode="niche"` it prioritises specialist fits. Scores are information-theoretic (PMI-match, IDF-rare boost, absence-bonus, Monty Hall opposition sharpening, MMR diversification, all combined submodularly) — see [v0.4.1 release notes](https://github.com/rohanvinaik/ModelAtlas/releases/tag/v0.4.1) for the full scoring layer.

---

## Three levels of comparison

All queries run against both systems. HuggingFace uses its API with `pipeline_tag` filters + sort-by-likes. ModelAtlas uses `navigate_models` with `quality=+1`. All results are real.

### Level 1: ModelAtlas matches HuggingFace

Common queries where HF works well. The baseline test — *can ModelAtlas reproduce the known-good answers?*

| Query | HuggingFace | ModelAtlas |
|-------|-------------|-----------|
| Sentiment analysis | cardiffnlp/twitter-roberta-sentiment ✓ | **Same model** + ProsusAI/finbert (financial sentiment) |
| Named entity recognition | dslim/bert-base-NER ✓ | **Same model** + microsoft/deberta-v3-base |
| Image captioning | Salesforce/blip-captioning-large ✓ | OpenGVLab/InternVL2-2B, Qwen2-VL-2B ✓ |

**Both systems return the right models.** This is the most important result. Anyone can build a niche search tool. Building one that also matches the incumbent beat-for-beat on common queries is what makes it a replacement, not a toy.

### Level 2: ModelAtlas exceeds HuggingFace

Queries with direction ("small"), intent ("fast"), or domain specificity ("medical classifier") — concepts that don't map to a single HF tag.

| Query | HuggingFace | ModelAtlas |
|-------|-------------|-----------|
| Small code model | codeparrot-small (33 likes, from 2021) | Qwen2.5-Coder-0.5B-Instruct (official, high-downloads) |
| Fast embedding model | *No results* — "fast" isn't a tag | Qwen3-Embedding-0.6B, jina-v5-text-small (sub-1B, edge-deployable) |
| Medical classifier | medical_o1_verifier (a *verifier*, not a classifier) | StanfordAIMI/stanford-deidentifier-base, obi/deid_bert_i2b2 |

HuggingFace starts returning noise. "Small" matches models with "small" in the name. "Fast" returns nothing. "Medical classifier" returns a reasoning verifier. ModelAtlas returns what you *meant*, not what you *typed*.

### Level 3: ModelAtlas finds the unfindable

Multi-constraint queries with direction + domain + negation. HuggingFace cannot express these at all.

| Query | HuggingFace | ModelAtlas |
|-------|-------------|-----------|
| Multilingual chat, NOT code/math/embedding | *Impossible to express* | PaddleOCR-VL-1.5 (sub-1B), Nanbeige4.1-3B-GGUF |
| Tiny on-device TTS | *No results* | **MioTTS-0.1B** (100M params), CosyVoice3-0.5B |
| Biology classifier, encoder-only | *No results* | BiomedBERT, gliner-biomed, **PoetschLab/GROVER** (genomics) |
| Small finance classifier | *No results* — "finance" isn't a pipeline tag | **FutureMa/Eva-4B** (finance+classification, trending), DMindAI/DMind-3-mini |
| Distilled reasoning, sub-3B, NOT a fine-tune | *No results* | Qwen3.5-0.8B-Opus-Reasoning-Distilled (score: 1.0) |

A 100-million-parameter TTS model. A genomics classifier with 7 anchors. A 0.8B model distilled from Claude Opus. These models exist on HuggingFace but they are **invisible** to keyword search. ModelAtlas finds them because `biology-domain + classification + encoder-only` is a precise intersection in a coordinate system, not a string match.

**The pattern:** Simple queries → both work. Directional queries → MA wins. Multi-constraint queries → HF returns nothing; MA finds exactly what you need. The harder the question, the wider the gap.

---

## What the LLM gets

This is an MCP tool. An LLM calls it during conversation. One tool call returns:

```json
{
  "model_id": "ibm-granite/granite-3b-code-instruct-128k",
  "score": 0.86,
  "score_breakdown": {"bank_alignment": 1.0, "anchor_relevance": 0.86, "coherence": 1.0},
  "positions": {"CAPABILITY": "+3", "EFFICIENCY": "-1", "DOMAIN": "+1"},
  "anchors": ["code-generation", "tool-calling", "long-context", "math", "consumer-GPU-viable"]
}
```

From this, the LLM *immediately knows*: small, code-focused, tool-calling, math-capable, consumer hardware, 128K context. The anchors are a vibe. The positions are a profile. The score explains *why this model and not another.*

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

**Scoring:** `bank_alignment × anchor_relevance × seed_similarity × coherence`. Multiplicative — a model that nails efficiency but misses capability gets zero, not fifty percent. Wrong-direction models decay hyperbolically. Avoided anchors stack exponentially (each halves the score). Required anchors are hard filters. The `coherence` factor comes from the certifier's per-model verification (below). The result is a scoring surface that strongly favors precise matches and rapidly eliminates mismatches, without binary cutoffs. [Full scoring math →](docs/DESIGN.md)

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
                                   certify()      ── 45 declarative Rule objects
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

45 rules across 6 categories cover the common structural implications: `pipeline_tag`, `model_type`, `library_name`, `quantization_level`, `safetensors`, tag conventions, and family-lineage evidence.

Rule tiers correspond to evidence trust:

- **Tier 1 (STRUCTURAL)** — the trigger is an HF-published field. Contradictions from a Tier-3 emission (LLM inference, web scrape) get REJECTED. Non-negotiable.
- **Tier 2 (SEMI-STRUCTURAL)** — tag conventions, family word in the repo name. Contradictions get DEMOTED (confidence lowered).
- **Tier 3 (INFERRED)** — advisory. Contradictions emit a WARNING; the emission survives.

**Result**: every model in the corpus carries a `certification_score` metadata field. 99.3% of models score ≥ 0.99 (no contradictions surfaced). The 0.7% with sub-perfect scores are surfaced in `navigate_models` results as a soft ranking tiebreaker — coherent evidence ranks above internally-contradictory evidence when all other constraints match.

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
| `navigate_models` | **Primary.** Bank directions + anchor constraints + context anchors + EPA target + `mode` + `bank_weights` → scored, ranked results (coherence-weighted, PageRank-boosted, Monty-Hall-sharpened, MMR-diversified, tie-clusters named). See [v0.4.1 scoring layer](https://github.com/rohanvinaik/ModelAtlas/releases/tag/v0.4.1). |
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

**50,906 models. 192 anchors. 377K model-anchor links across 8 banks. 238K signed positions. 99.6% bank coverage. 4,027 models web-enriched. 45 declarative rules enforced at every write. 100% of models certifier-scored (99.3% at perfect coherence).**

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
