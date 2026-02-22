# hf-model-search

MCP server that builds a **navigable semantic network** of ML models. Models are positioned across orthogonal semantic banks (architecture, capability, efficiency, compatibility, lineage, domain, quality) and connected through a shared anchor vocabulary. Queries are navigational — find models by exploring semantic space, not just filtering columns.

## Quick Start

```bash
# Install and run (uv required)
cd /Users/rohanvinaik/tools/infrastructure/hf-model-search
uv sync

# Already registered as global MCP server in ~/.claude.json
# Available in every Claude Code session automatically
```

## Why This Exists

HuggingFace exposes ~10 structured fields per model. This tool extracts and encodes **relational and hierarchical information** — architecture type, capability profiles, model lineage, efficiency characteristics, domain specialization — into a structured semantic space that supports queries HF can't:

- "Most general Llama base model that supports tool-calling and runs on consumer GPU"
- "Models architecturally similar to Mamba but with instruction tuning"
- "What's the fine-tune lineage of this model? What siblings does it have?"
- "Navigate from this model toward smaller and more code-focused"

The point isn't to rank models by some objective "usefulness" metric. It's to see **what exists** that can be worked into your projects — finding the best model, or the most general version of some capability to fine-tune.

## Architecture

### The Semantic Network (Primary Storage)

Models live in a 7-bank semantic space, inspired by the [Sparse Wiki Grounding](https://github.com/rohanvinaik/sparse-wiki-grounding) architecture. Each bank is an orthogonal dimension with a meaningful zero state:

```
ARCHITECTURE    zero: standard transformer decoder
                -N ← simpler/older          +N → novel/specialized (Mamba, SSM, MoE)
                ~95% of filtering happens in this bank alone

CAPABILITY      zero: general language model
                -N ← narrow/single-task     +N → rich (code, reasoning, tool-calling, NER)

EFFICIENCY      zero: ~7B mainstream
                -N ← tiny (1B, 0.5B)        +N → massive (30B, 70B, frontier)

COMPATIBILITY   zero: standard transformers + PyTorch
                                             +N → specific format/framework/hardware

LINEAGE         zero: base/foundational model of a family
                -N ← predecessors/ancestors  +N → fine-tunes, quantizations, derivatives

DOMAIN          zero: general knowledge
                                             +N → increasingly specialized (code → Python → DSL)

QUALITY         zero: established, mainstream
                -N ← legacy/abandoned        +N → trending, high momentum
```

### Anchor Dictionary (Emergent Similarity)

A shared vocabulary of characteristics ("instruction-following", "Apple-Silicon-native", "RLHF-tuned", "tool-calling") that models link to. Models sharing anchors cluster together without explicit edges. Anchor sets support set operations:

- **Intersection**: what two models have in common
- **Symmetric difference**: what distinguishes them
- **Jaccard similarity**: overall semantic overlap

### SQL Overflow Table (Sidecar)

Flat metadata that doesn't decompose into the network: SHA hashes, exact dates, specific benchmark scores, download count snapshots, license strings. In a perfect design this table would be empty — in practice it catches the non-relational overflow.

### Full Architecture Diagram

```
Source Adapters (HF, Ollama, Replicate, CivitAI)
    │
    ▼
Extraction Pipeline
    │  reads model cards, configs, API metadata, file lists
    │  produces: bank positions, anchor links, model relationships, overflow metadata
    │
    ▼
┌─────────────────────────────────────────────────────┐
│              Semantic Network (SQLite)                │
│                                                       │
│  models ─── model_positions (7 banks, signed)         │
│         ─── model_anchors ─── anchors (dictionary)    │
│         ─── model_links (lineage, family, variants)   │
│         ─── model_metadata (overflow sidecar)         │
│                                                       │
│  Query: navigational (bank-aware similarity,          │
│         anchor set operations, lineage traversal)     │
└─────────────────────────────────────────────────────┘
    │
    ▼
MCP Server (tools for search, navigate, compare, index)
```

## Tools

### `hf_search_models`

Primary navigational search. Combines bank-position constraints with anchor similarity.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | *required* | Natural language search query |
| `task` | str \| None | None | Pipeline tag filter (maps to ARCHITECTURE/CAPABILITY banks) |
| `author` | str \| None | None | Filter by org |
| `library` | str \| None | None | Filter by library (maps to COMPATIBILITY bank) |
| `min_likes` | int | 0 | Minimum likes (QUALITY bank signal) |
| `limit` | int | 20 | Results to return |

### `hf_build_index`

Fetch models from a source, run through extraction pipeline, add to the semantic network. Additive — multiple calls enrich the same network.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | str \| None | None | Scope what gets fetched from source API |
| `author` | str \| None | None | Scope by author/org |
| `limit` | int | 2000 | Max models to fetch |
| `min_likes` | int | 5 | Noise filter |
| `force` | bool | False | If true, clear and rebuild from scratch |

### `hf_get_model_detail`

Deep dive on one model. Returns full network position (all 7 banks), anchor set, lineage links, and overflow metadata.

### `hf_compare_models`

Compare models via anchor set operations. Shows shared anchors (intersection), distinguishing features (symmetric difference), and per-bank position comparison.

### `hf_index_status`

Network stats: total models, breakdown by bank positions, anchor dictionary size, source coverage.

## Storage

| Path | Contents |
|------|----------|
| `~/.cache/hf-model-search/network.db` | The semantic network (SQLite) |
| `~/.cache/hf-model-search/extraction_cache/` | Cached raw API responses (for re-extraction) |

## Dependencies

- `mcp[cli]` — MCP server framework
- `huggingface-hub` — HF API client
- `sentence-transformers` — Small local model for NLP extraction tasks
- `rapidfuzz` — Fuzzy string matching (name resolution layer)
- `sqlite3` — Standard library, primary storage

## Design Reference

Full system design: `.claude/CLAUDE.md`
Theoretical foundation: `/Users/rohanvinaik/sparse-wiki-grounding/reports/ARCHITECTURE_DEEP_DIVE.md`
