---
generated: true
generated_from: 
  - docs/DESIGN.md
source_hash: 42e35429474ea311
spec_hash: 1e5142b9115a095c
file_hash: d80d50c1a8a323ae
materializer_version: "0.1.0"
theory_scope: false
audience: operator
page_id: architecture-server
---

# Architecture: MCP Server & Query Engine


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
