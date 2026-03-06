---
generated: true
generated_from: 
  - docs/wiki/query-engine.md
source_hash: c389a1fdb0e9ed4d
spec_hash: 41ac3796d7230dc1
file_hash: da38809642daa468
materializer_version: "0.2.0"
theory_scope: false
audience: operator
page_id: query-engine
---

# Query Engine


# Query Engine

**`navigate_models` scores the entire model space with three multiplicative signals — bank alignment, anchor relevance, and seed similarity — using four indexed SQL queries. Every score traces back to named components. Nothing is a black box.**

---

## Problem Space

A scoring function for model discovery has competing requirements. It must handle multiple dimensions simultaneously (efficiency AND capability AND compatibility). It must express direction, not just match ("toward smaller," not "is small"). It must be transparent — users need to understand why a result ranked where it did.

Averaging across dimensions fails: a model that perfectly matches capability but completely fails efficiency scores 0.5, which looks plausible but isn't. Embedding similarity is opaque. Boolean filters are too rigid.

`navigate_models` uses multiplicative scoring to solve all three.

---

## Core Model

### The Three Signals

```
final_score = bank_alignment × anchor_relevance × seed_similarity
```

Each component is in [0, 1]. The product means any zero kills the score. This is deliberate — see [Navigation Geometry](Navigation-Geometry) for why.

### Bank Alignment

For each [bank](Glossary#bank) where the caller specifies a direction (-1, 0, or +1):

| Condition | Score | Reasoning |
|-----------|-------|-----------|
| Direction = 0 | `1 / (1 + |position|)` | Want zero: penalize distance from origin |
| Model aligned with direction | `1.0` | On the right side |
| Model at zero | `0.5` | Neutral — not wrong, not aligned |
| Model opposed | `1 / (1 + |alignment|)` | Wrong side: decay with distance |

Scores are **multiplied** across all specified banks. Omitted banks contribute 1.0 (neutral). Missing bank position on a model → 0.3 penalty.

### Anchor Relevance

Three [anchor](Glossary#anchor) lists with different treatments:

**require** — Hard SQL pre-filter. Models must have ALL required anchors. Missing any → excluded before scoring. This runs as `SELECT ... HAVING COUNT(DISTINCT anchor_id) = N`, eliminating non-candidates at the database level.

**prefer** — [IDF-weighted](Glossary#idf-weighting) overlap:
```
score = sum(IDF[matched]) / sum(IDF[all preferred])
```
Rare anchors count more. `proof-assistant` (12 models) contributes ~50x more than `decoder-only` (17K models). This falls directly out of information theory.

**avoid** — Penalty. Each avoided anchor present halves the score:
```
score = 0.5 ^ count_of_avoided_present
```

### Seed Similarity

When `similar_to` specifies a model, IDF-weighted Jaccard between the seed's anchor set and each candidate's:

```
similarity = sum(IDF[shared anchors]) / sum(IDF[all unique anchors])
```

Standard Jaccard treats all anchors equally. IDF weighting means sharing a rare anchor matters more than sharing a common one. See [Emergent Similarity](Emergent-Similarity).

---

## Evidence: The Four Queries

The entire scoring pipeline executes four SQL queries:

1. **Pre-filter** on required anchors — `SELECT model_id FROM model_anchors WHERE anchor_id IN (...) GROUP BY model_id HAVING COUNT(DISTINCT anchor_id) = N`
2. **Batch positions** — one query loads all candidate [signed positions](Glossary#signed-position) across all banks
3. **Batch anchors** — one query loads all candidate anchor sets
4. **Batch authors** — one query loads display metadata

Then: scoring is arithmetic on in-memory Python dicts. No per-model queries. No N+1 patterns.

**Performance consequence:** Four indexed queries instead of 18K individual `get_model()` calls. The database does set operations; Python does multiplication.

### IDF Cache

IDF values are computed once per anchor at module load: `log(N / count)` where N is total model count. Cached at module level. Invalidated after index builds.

---

## What This Is Not

- **Not a recommendation engine.** No user modeling, no collaborative filtering. Scoring is a pure function of model properties and query parameters.
- **Not approximate.** Every model in the candidate set is scored exactly. There's no ANN index, no sampling, no top-K approximation before scoring.
- **Not the only query path.** `hf_search_models` provides a keyword-based fallback with weighted-sum scoring (not multiplicative). It works for quick lookups but has known limitations: no IDF weighting, N+1 queries, no negative constraints.

---

## Related Concepts

- [Bank alignment](Glossary#bank-alignment) — how positions become scores
- [IDF weighting](Glossary#idf-weighting) — the information-theoretic foundation
- [Navigation Geometry](Navigation-Geometry) — why multiplicative, why movement
- [Signed Hierarchies](Signed-Hierarchies) — the coordinate system being scored against

---

*[← System Overview](System-Overview) · [Extraction Pipeline →](Extraction-Pipeline)*
