# Data Model

**Six tables, eight [banks](Glossary#bank), ~170 [anchors](Glossary#anchor), and a flat metadata overflow. Everything queryable. Everything with provenance.**

---

## Problem Space

A database for model discovery could go two ways. A wide table with 65 columns — one for each attribute anyone might want. Or a normalized relational model that separates structure from values. The first is rigid (every new attribute requires a schema migration) and sparse (most models don't have most attributes). The second is flexible but can become a schema archaeology project.

ModelAtlas takes a third path: a **small fixed schema** built around the semantic primitives ([banks](Glossary#bank), [anchors](Glossary#anchor), [links](Glossary#anchor)) with a flat overflow table for everything else. The schema hasn't changed since inception because the primitives are general enough to absorb new information without new tables.

---

## Core Model

### Schema

```sql
models(model_id PK, author, source, display_name)

model_positions(model_id, bank)
    → sign, depth, path_nodes, zero_state

model_links(source_id, target_id, relation)
    → weight

anchors(anchor_id PK, label UNIQUE, bank, category, source)

model_anchors(model_id, anchor_id)
    → weight, confidence

model_metadata(model_id, key)
    → value, value_type
```

### What Each Table Does

**models** — One row per model. The `model_id` is the HuggingFace-style `author/name` identifier. `source` tracks where the model was fetched from (huggingface, ollama, stub).

**model_positions** — [Signed positions](Glossary#signed-position) in each bank. A model can have positions in any subset of the eight banks. `path_nodes` stores the hierarchical labels for models that have sub-categories (e.g., `["decoder-only", "causal-lm"]`).

**model_links** — Explicit model-to-model relationships:

| Relation | Weight | Meaning |
|----------|--------|---------|
| `fine_tuned_from` | 0.9 | Derivative via fine-tuning |
| `quantized_from` | 0.85 | Quantized variant |
| `variant_of` | 0.8 | Official size/instruct variant |
| `same_family` | 0.7 | Shares a model family |
| `predecessor` | 0.6 | Earlier generation |
| `successor` | 0.6 | Later generation |

**anchors** — The [anchor dictionary](Glossary#anchor). Each anchor has a unique label, belongs to one bank, and tracks its `source` provenance (bootstrap, deterministic, pattern, vibe, expansion).

**model_anchors** — The many-to-many link between models and anchors. `weight` scales relevance. `confidence` reflects extraction certainty (1.0 for deterministic, 0.5 for LLM-assigned).

**model_metadata** — [Overflow](Glossary#overflow-metadata). Key-value pairs for everything that doesn't fit the network structure: SHA hashes, creation dates, raw parameter counts, license strings, benchmark scores, [vibe summaries](Glossary#vibe-summary), quality scores.

### Indices

```sql
model_positions(bank)           -- "all models' EFFICIENCY positions"
model_links(source_id)          -- "what links FROM this model"
model_links(target_id)          -- "what links TO this model"
model_anchors(model_id)         -- "all anchors for this model"
model_anchors(anchor_id)        -- "all models with this anchor"
model_metadata(key)             -- "all models with quality_score"
```

The bidirectional anchor index is critical — "what models share this anchor?" must be fast for [IDF-weighted](Glossary#idf-weighting) similarity computation.

---

## Evidence: Scale

| Metric | Count |
|--------|-------|
| Models | 19,498 |
| Bank positions | ~80,000 |
| Anchors | 166 |
| Model-anchor links | 128,000+ |
| Model-to-model links | ~2,000 |
| Metadata entries | ~60,000 |
| Database size | ~80 MB |

The entire semantic network fits in a single SQLite file. No external services at query time.

### Provenance Tables (Phase D)

Three additional tables track the audit and correction pipeline:

```sql
phase_d_runs(run_id PK, phase, status, config_json, summary_json)
audit_findings(model_id, run_id, finding_type, details_json)
correction_events(model_id, run_id, original_json, healed_json, rationale)
```

Every D-phase operation has a UUID, every finding has a type (contradiction, gap, confidence_conflict), and every correction preserves the original for training data export.

---

## What This Is Not

- **Not a wide table.** Eight banks replace 65 columns. New model attributes don't require schema changes.
- **Not a graph database.** SQLite with relational indices. Graph-like traversal happens in Python from batch-loaded data, not in a graph query language.
- **Not append-only.** Positions, anchors, and metadata are updated in place during healing passes. Provenance is tracked in the correction tables, not in the main tables.

---

## Related Concepts

- [Signed position](Glossary#signed-position) — what positions mean
- [Anchor](Glossary#anchor) — the shared vocabulary
- [Extraction Pipeline](Extraction-Pipeline) — how the data gets in
- [Data Distribution](Data-Distribution) — how the database gets to users

---

*[← Extraction Pipeline](Extraction-Pipeline) · [Data Distribution →](Data-Distribution)*
