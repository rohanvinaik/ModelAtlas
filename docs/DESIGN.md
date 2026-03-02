# ModelAtlas — Theory and Design

## 1. Core Thesis

HuggingFace Hub has ~1M models and exposes ~10 structured fields per model. The relational and hierarchical information that makes models *findable* — capabilities, lineage, efficiency profiles, architectural properties, domain specializations — is trapped in unstructured model cards, naming conventions, config files, and community knowledge.

ModelAtlas extracts that information and encodes it as **positions in a structured semantic space**. The result is a navigable network where queries are movements through space ("toward smaller and more code-focused"), not WHERE clauses ("parameter_count < 3B AND tags LIKE '%code%'").

The theoretical foundation comes from the Sparse Wiki Grounding architecture: signed hierarchies with zero states, orthogonal semantic banks, and anchor dictionaries that produce emergent similarity through set overlap.

## 2. Data Model

### 2.1 Semantic Banks

Eight orthogonal dimensions. Each has a **zero state** placed at the query-frequency mode — the thing most people are looking for. Signed distance from zero encodes direction: negative toward general/simpler/earlier, positive toward specialized/derived/novel.

```
Bank            Zero State                  Negative              Positive
─────────────── ─────────────────────────── ───────────────────── ──────────────────────
ARCHITECTURE    transformer decoder         encoder-only, RNN     Mamba, RWKV, SSM, MoE
CAPABILITY      general language model       single-task, embed    code, tools, reasoning
EFFICIENCY      ~7B parameters              sub-1B, 1B, 3B       13B, 30B, 70B, frontier
COMPATIBILITY   PyTorch + transformers       —                    GGUF, MLX, Apple Silicon
LINEAGE         base/foundational model     predecessors          fine-tune, quant, merge
DOMAIN          general knowledge            —                    code, medical, legal
QUALITY         established mainstream       legacy, abandoned    trending, rising
TRAINING        standard supervised (SFT)   LoRA, distillation    RLHF, DPO, multi-stage
```

A model's position is stored as `(sign, depth)` — e.g. `(-1, 2)` means two steps in the negative direction. Signed position = `sign * depth`.

**Why signed hierarchies instead of flat categories:** A categorical "size" field with values {small, medium, large} can't express proximity. Signed positions give gradient scoring — a 3B model is *close* to 7B, not a binary mismatch. And zero placement at the mode means most queries resolve near the origin with minimal computation.

### 2.2 Anchor Dictionary

A shared vocabulary of ~130+ semantic labels. Each anchor belongs to one bank, and models link to the anchors that describe them.

Example anchors by bank:

| Bank | Anchors |
|------|---------|
| ARCHITECTURE | `transformer`, `mamba`, `mixture-of-experts`, `diffusion`, `vision-transformer` |
| CAPABILITY | `instruction-following`, `tool-calling`, `code-generation`, `reasoning`, `multimodal` |
| EFFICIENCY | `7B-class`, `sub-1B`, `quantized`, `consumer-GPU-viable`, `edge-deployable` |
| COMPATIBILITY | `GGUF-available`, `MLX-compatible`, `Apple-Silicon-native`, `vLLM-compatible` |
| LINEAGE | `Llama-family`, `Mistral-family`, `Qwen-family`, `base-model`, `fine-tune` |
| DOMAIN | `code-domain`, `medical-domain`, `legal-domain`, `Python-code`, `math-domain` |
| QUALITY | `trending`, `high-downloads`, `community-favorite`, `official-release`, `high-mmlu`, `strong-humaneval` |
| TRAINING | `sft-trained`, `rlhf-trained`, `dpo-trained`, `lora-adapted`, `distilled`, `trained-on-synthetic-data` |

Anchors create **emergent similarity**. Two models sharing 15 anchors are semantically related without an explicit edge. The anchor dictionary *is* the semantic vocabulary of the system.

Anchors grow organically. New ones are minted during extraction (Tier 2 pattern matching and Tier 3 vibe extraction). Each anchor tracks its provenance: `bootstrap`, `deterministic`, `pattern`, or `vibes`.

### 2.3 Explicit Links (Layer 1)

Direct model-to-model relationships:

| Relation | Weight | Meaning |
|----------|--------|---------|
| `fine_tuned_from` | 0.9 | Derivative via fine-tuning |
| `quantized_from` | 0.85 | Quantized variant |
| `variant_of` | 0.8 | Official size/instruct variant |
| `same_family` | 0.7 | Shares a model family |
| `predecessor` | 0.6 | Earlier generation |
| `successor` | 0.6 | Later generation |

### 2.4 Overflow Metadata

A flat `(model_id, key, value)` table for data that doesn't decompose into the network: SHA hashes, creation dates, raw parameter counts, license strings, benchmark scores. In a perfect design this table would be empty. In practice it catches the long tail.

The one exception is `vibe_summary` — a single prose sentence per model capturing its irreducible "feel." This is the *only* natural language stored.

### 2.5 Database Schema

```sql
models(model_id PK, author, source, display_name)
model_positions(model_id, bank) → sign, depth, path_nodes, zero_state
model_links(source_id, target_id, relation) → weight
anchors(anchor_id PK, label UNIQUE, bank, category, source)
model_anchors(model_id, anchor_id) → weight, confidence
model_metadata(model_id, key) → value, value_type
```

Indices on `positions(bank)`, `links(source_id)`, `links(target_id)`, `model_anchors(model_id)`, `model_anchors(anchor_id)`, `metadata(key)`. The bidirectional anchor index is critical — "what models share this anchor?" must be fast.

## 3. Extraction Pipeline

Three tiers of increasing complexity extract structure from raw model data.

### Tier 1: Deterministic

Pure arithmetic on structured API fields. No ambiguity.

- `parameter_count` → EFFICIENCY sign and depth (log-scale bucketing: <1B=-2, 1-3B=-1, 3-10B=0, 10-35B=+1, 35-100B=+2, >100B=+3)
- `architecture_type` from config.json → ARCHITECTURE position
- `downloads`, `likes`, `created_at` → QUALITY position (velocity-aware: high recent downloads on a new model = trending)
- `license`, `sha`, `context_length` → overflow metadata

**Input:** `ModelInput` dataclass (model_id, author, tags, pipeline_tag, parameter_count, config, etc.)
**Output:** `DeterministicResult` with `BankPosition` objects and metadata key-value pairs.

### Tier 2: Pattern Matching

Regex and heuristics on tags, model names, file lists, and config structure.

- Model ID contains "instruct"/"chat" → `instruction-following`/`chat` anchors + CAPABILITY position
- Tags include "gguf"/"gptq"/"awq" → compatibility anchors + COMPATIBILITY position
- `base_model` field present → `fine_tuned_from` link + LINEAGE position
- Pipeline tag "text-generation" + tags "code" → `code-generation` anchor + DOMAIN position
- File list contains `.gguf` files → `GGUF-available` anchor
- Training methodology signals (RLHF, DPO, LoRA, SFT, distillation, etc.) → training anchors + TRAINING position
- Training dataset detection from tags (`dataset:*`) and keyword matching → `training_datasets` metadata

**Output:** `PatternResult` with additional anchors, positions (including TRAINING bank), and explicit links.

### Tier 3: Vibe Extraction

LLM-based structured generation for what symbolic parsing can't capture. Uses Outlines with Pydantic schemas to constrain a small local model (Qwen2.5-0.5B-Instruct) into producing:

- `vibe_summary`: one sentence capturing the model's distinctive character
- `extra_anchors`: capability signals detected from model card prose

Tier 3 only runs on models above a quality threshold (default: 50+ likes). Anchors from vibes are stored with `confidence=0.5` to reflect their heuristic origin.

### Pipeline Orchestration

`extract_and_store()` runs all three tiers in sequence on a single model, writing results to the network database. `extract_batch()` processes lists. The pipeline is idempotent — re-extracting a model upserts its positions and anchors.

## 4. Query Engines

### 4.1 `navigate_models` — Structured Scoring (Primary)

The calling LLM decomposes any natural language query into structured parameters. ModelAtlas does deterministic math.

**Three signals, multiplicative:**

```
final_score = bank_alignment * anchor_relevance * seed_similarity
```

Each component is in [0, 1]. The product means any zero kills the score — a model that perfectly matches efficiency but completely fails capability gets 0, not 0.5.

#### Bank Alignment

For each bank where the caller specified a direction (-1, 0, or +1):

```python
if direction == 0:   score = 1 / (1 + |position|)        # want zero: penalize distance
if aligned:          score = 1.0                           # on the right side
if at zero:          score = 0.5                           # neutral
if opposed:          score = 1 / (1 + |alignment|)        # wrong side: decay
```

**Multiplicative** across all specified banks. Omitted banks don't contribute (neutral 1.0). Missing bank position on a model → 0.3 penalty.

#### Anchor Relevance (IDF-weighted)

Three anchor lists with different treatments:

- **require**: Hard SQL pre-filter. `SELECT model_id ... HAVING COUNT(DISTINCT anchor_id) = N`. Missing any required anchor → model excluded before scoring.
- **prefer**: IDF-weighted overlap. `score = sum(idf[matched]) / sum(idf[all_preferred])`. Rare anchors (e.g. "proof-assistant" on 12 models) count far more than ubiquitous ones ("decoder-only" on 17K models).
- **avoid**: Penalty. Each avoided anchor present halves the score: `0.5 ^ count`.

IDF = `log(N / count_models_with_anchor)`, computed once and cached at module level. Invalidated after index builds.

#### Seed Similarity

When `similar_to` is specified, IDF-weighted Jaccard between seed's anchor set and candidate's:

```
idf_intersection / idf_union
```

Standard Jaccard treats all anchors equally. IDF weighting means sharing a rare anchor ("proof-assistant") matters more than sharing a common one ("decoder-only").

#### Performance

Batch SQL replaces N+1 per-model lookups:

1. Pre-filter candidates by required anchors (SQL `HAVING`)
2. `batch_get_positions()` — one query for all candidate positions
3. `batch_get_anchor_sets()` — one query for all candidate anchors
4. `batch_get_authors()` — one query for display info
5. Score in Python from in-memory dicts

Four indexed queries instead of 18K individual `get_model()` calls.

### 4.2 `hf_search_models` — Natural Language Search (Fallback)

Parses natural language via keyword dictionaries, combines four scoring channels with weighted sum:

| Channel | Weight | Source |
|---------|--------|--------|
| Bank proximity | 0.30 | Gradient scoring against parsed `BankConstraint`s |
| Anchor overlap | 0.35 | Confidence-weighted Jaccard on target anchor labels |
| Spreading activation | 0.15 | Bellman-Ford propagation from seed models |
| Fuzzy matching | 0.20 | RapidFuzz token ratio on model IDs and tags |

This is the original query engine. It works for quick keyword searches but has known limitations: no IDF weighting, averaged (not multiplicative) bank scores, no negative constraints, and N+1 per-model lookups. `navigate_models` fixes all of these.

### 4.3 Spreading Activation

Priority-queue Bellman-Ford propagation from seed models. Two channels:

- **Link channel (Layer 1):** Traverse `model_links` bidirectionally with relation-specific weights (fine_tuned_from: 0.9, same_family: 0.7, etc.)
- **Anchor channel (Layer 2):** Find models that share anchors with the current node, weighted by fraction of shared anchors.

Activation decays by 0.8 per hop, maximum 3 hops. Bank scoping prevents semantic bleeding — if the query is about CAPABILITY, spreading only traverses CAPABILITY-bank anchors.

Used by `hf_search_models` for "models like X" queries. Not used by `navigate_models` (IDF-weighted Jaccard on seed anchors captures the same signal more efficiently for the structured case).

### 4.4 Supporting Queries

- **`compare(model_ids)`** — Set operations on anchor sets (intersection, symmetric difference, Jaccard) plus per-bank position deltas.
- **`similar_to(model_id)`** — Vanilla Jaccard similarity against all models (used by the old NL engine).
- **`lineage(model_id)`** — Traverse LINEAGE bank: predecessors, base, derivatives, family members, ordered by signed position.

## 5. Source Adapters

Pluggable fetchers behind a common `SourceAdapter` protocol. All produce the same `ModelInput` for the extraction pipeline.

| Source | Status | Data Quality |
|--------|--------|-------------|
| HuggingFace | Primary, complete | Richest — API fields, model cards, configs, file lists |
| Ollama | Working | Hardware compat info, simpler metadata, local-only |
| Replicate | Future | API-oriented, pricing, latency |
| CivitAI | Future | Image/diffusion, community ratings |

Source-agnostic: the same model from HF and Ollama occupies the same point in semantic space and links to the same anchors. The `source` field on models tracks provenance.

## 6. Ingestion

### Background Daemon

Multi-phase pipeline run by `model-atlas-ingest`:

| Phase | What | Cost |
|-------|------|------|
| A: Fetch | Stream HF API sorted by likes, cache raw JSON + config.json + model cards | Network I/O |
| B: Extract | Tier 1+2 (deterministic + patterns), populate network | CPU only |
| C: Intelligence | Multi-tier extraction (see below) | CPU-only inference |

### Phase C: Multi-Tier Intelligence Extraction

Research shows model cards vary wildly in completeness (74.3% have training info, only 17.4% have limitations, 2% environmental impact — [arxiv 2402.05160](https://arxiv.org/html/2402.05160v1)). A one-size-fits-all LLM prompt can't adapt to this variance. Phase C uses specialized models for what they're best at, ground truth datasets for calibration, and a quality gate to catch artifacts.

```
C1: Smol-Hub-tldr (360M) ─── card_text models ──► smol_summary
C2: qwen2.5:3b (Ollama) ─── all models ─────────► qwen_summary + extra_anchors
    [C1 and C2 run in parallel]
         │
         ▼
Summary Selection ─── smol preferred when available ──► vibe_summary
         │
         ▼
C3: Quality Gate ─── blind review (summary+anchors only) ──► quality_score
         │
         ▼
C4: Ground Truth ─── offline comparison vs reference datasets
```

| Sub-phase | Model | Input | Output | Runtime |
|-----------|-------|-------|--------|---------|
| C1 | davanstrien/Smol-Hub-tldr (360M) | card_text | smol_summary | ~1 hour (6.7K models) |
| C2 | qwen2.5:3b (Ollama) | enriched prompts | qwen_summary + anchors | ~3 days (38K models, 2 machines) |
| C3 | qwen2.5:3b (Ollama) | summary + anchors only | quality_score (0-1) | ~3 days |
| C4 | None (offline) | our outputs vs reference | similarity metrics | seconds |

**Model selection rationale:** Smol-Hub-tldr (360M) was SFT'd specifically on Llama 3.3 70B distillations of HF model cards — it produces better summaries than qwen2.5:3b for models with rich card text, but can't generate anchors or handle models without cards. qwen2.5:3b handles structured extraction (summary + anchors) for all models regardless of card availability.

**Quality gate design:** C3 is a blind review — it sees only the generated summary and anchors, NOT the original source material. This prevents the reviewer from simply restating the source. Three axes rated 0-3 (specificity, coherence, artifacts), threshold 0.5 (4.5/9.0).

**Ground truth calibration:** C4 compares against two curated datasets (davanstrien/hub-tldr-model-summaries-llama for summaries, davanstrien/parsed-model-cards for structured fields) using string similarity and anchor coverage metrics.

**Extended corpus (C1b):** Beyond our ~7K enriched models, C1 can process the full HuggingFace corpus by streaming from librarian-bots/model_cards_with_metadata (975K models, daily-refreshed). Card text is transient — never stored on disk, only model_id + summary persists.

All workers are standalone scripts with zero ModelAtlas imports, deployable via scp to any machine with Python. See [`docs/pipeline.md`](pipeline.md) for the full operational reference.

### Seed Strategy

Multi-pass HF streaming with client-side filters:

1. **Core:** top models by likes (100+ threshold)
2. **Expand:** broader coverage (10+ likes, post-2023)
3. **Niche:** targeted categories — code, medical, legal, multilingual (3+ likes)

### Update Model

Model data is mostly static. Parameter counts, architectures, and training data don't change post-release. The real "update" is discovering and adding new models, not modifying existing entries.

- Periodic additive sweeps: find new models above thresholds, run extraction, INSERT
- Download/like snapshots: periodic captures for velocity (QUALITY signal)
- Lineage updates: new fine-tunes get links to their base models

## 7. Module Map

```
src/model_atlas/
├── server.py              MCP tool definitions (FastMCP)
├── query.py               Query engines: navigate(), search(), compare(), lineage()
├── query_types.py         Data classes: StructuredQuery, NavigationResult, SearchResult, ...
├── spreading.py           Bellman-Ford spreading activation
├── db.py                  SQLite schema, CRUD, batch queries, IDF computation
├── config.py              All constants: weights, thresholds, paths
├── cache.py               Disk-based model card caching with TTL
├── ingest.py              Three-phase background ingestion daemon
├── _formatting.py         JSON response formatters for MCP tools
├── extraction/
│   ├── deterministic.py   Tier 1: structured fields → positions + metadata
│   ├── patterns.py        Tier 2: regex/heuristics → anchors + links
│   ├── vibes.py           Tier 3: Outlines LLM → vibe_summary + extra anchors
│   └── pipeline.py        Orchestrator: extract_and_store(), extract_batch()
├── search/
│   ├── structured.py      Layer 1: HuggingFace Hub API search
│   └── fuzzy.py           Layer 2: RapidFuzz token ratio scoring
└── sources/
    ├── base.py            SourceAdapter protocol + SourceSearchResult
    ├── huggingface.py     HF Hub adapter
    ├── ollama.py          Local Ollama adapter
    └── registry.py        Source registration and lookup
```

## 8. Configuration

All tunable constants live in `config.py`:

| Constant | Value | Purpose |
|----------|-------|---------|
| `WEIGHT_BANK` | 0.30 | Bank proximity weight in NL search |
| `WEIGHT_ANCHOR` | 0.35 | Anchor Jaccard weight in NL search |
| `WEIGHT_SPREAD` | 0.15 | Spreading activation weight in NL search |
| `WEIGHT_FUZZY` | 0.20 | Fuzzy matching weight in NL search |
| `NAVIGATE_MISSING_BANK_PENALTY` | 0.30 | Score for missing bank position in navigate |
| `NAVIGATE_AVOID_DECAY` | 0.50 | Per-avoided-anchor multiplier in navigate |
| `SPREAD_DECAY` | 0.80 | Activation loss per hop |
| `SPREAD_MAX_DEPTH` | 3 | Maximum spreading hops |
| `DEFAULT_CANDIDATE_LIMIT` | 500 | HF API candidates per NL search |
| `DEFAULT_RESULT_LIMIT` | 20 | Results returned to caller |
| `INGEST_MIN_LIKES` | 5 | Minimum likes for Phase A/B |
| `INGEST_VIBE_MIN_LIKES` | 50 | Minimum likes for Phase C |
| `VIBE_MODEL_NAME` | Qwen/Qwen2.5-0.5B-Instruct | Local LLM for vibe extraction |

## 9. Invariants

1. **No embedded prose.** Model cards are inputs to extraction, not stored artifacts. Only `vibe_summary` is prose.
2. **Signed hierarchies on all banks.** Every bank has a zero state and models have signed distances from it.
3. **Anchors are the semantic vocabulary.** Emergent similarity through shared anchors, not opaque embeddings.
4. **Auditable scores.** Every similarity score traces back to specific shared anchors and bank positions.
5. **Source-agnostic.** Same model from different sources occupies the same point in semantic space.
6. **Network is primary, SQL is overflow.** If most data is in the metadata table, something is wrong.
7. **Multiplicative scoring in navigate.** A zero in any component kills the final score — no averaging away bad matches.
8. **IDF-weighted anchors in navigate.** Rare anchors count more than ubiquitous ones.
