---
generated: true
generated_from: 
  - docs/wiki/system-overview.md
source_hash: 689fac7a7c5772f2
spec_hash: dae094b67d844729
file_hash: 8f9408af8ae49543
materializer_version: "0.2.0"
theory_scope: false
audience: operator
page_id: system-overview
---

# System Overview


# System Overview

**The 30-second architecture: an MCP server that exposes a SQLite semantic network to LLMs. The LLM decomposes questions into coordinates and anchors. ModelAtlas does arithmetic and set intersections. Neither is smart alone, but the system is.**

---

## Problem Space

A model discovery system needs three things: a data layer that encodes structural relationships, a query engine that reasons about direction and proximity, and an interface that lets non-technical users ask structural questions without learning the schema.

ModelAtlas splits the intelligence across two systems. The LLM handles decomposition — turning "small code model with tool-calling" into structured parameters. ModelAtlas handles scoring — deterministic math on integers and sets. The MCP protocol is the bridge.

---

## Core Model

### System Map

```
┌─────────────────────────────────────────────────┐
│  User (natural language question)               │
│       ↓                                         │
│  LLM Client (Claude, etc.)                      │
│       ↓  decomposes into structured params      │
│  MCP Protocol                                   │
│       ↓                                         │
│  ModelAtlas MCP Server (9 tools)                 │
│       ↓                                         │
│  Query Engine (navigate_models)                  │
│       ↓  4 batch SQL queries                    │
│  SQLite Database (network.db)                    │
│       • 19K+ models                             │
│       • 8 banks × signed positions              │
│       • 170 anchors × 128K links                │
│       • explicit model-to-model links           │
│       • overflow metadata                       │
└─────────────────────────────────────────────────┘
```

### The MCP Tools

Nine tools exposed via the Model Context Protocol:

| Tool | Purpose | Rail |
|------|---------|------|
| `navigate_models` | Structured scoring with bank directions + anchors | Primary query |
| `hf_search_models` | Natural language keyword search (fallback) | Discovery |
| `hf_compare_models` | Anchor set operations + bank position deltas | Comparison |
| `hf_get_model_detail` | Full model profile with all positions and anchors | Inspection |
| `hf_build_index` | Fetch + extract models from HuggingFace | Data building |
| `hf_index_status` | Database statistics and coverage | Diagnostics |
| `search_models` | Fuzzy text search on model IDs | Quick lookup |
| `set_model_vibe` | Write/update vibe summary for a model | Curation |
| `list_model_sources` | Available data sources (HF, Ollama) | Configuration |

The calling LLM chooses the right tool based on the user's question. Most queries route through `navigate_models`. The LLM fills in the structured parameters; ModelAtlas returns scored results with explanations.

### Data Flow: Build vs Query

**Build time** (hours to days, runs once):
```
HuggingFace API → Phase A (fetch) → Phase B (extract) →
Phase C (LLM classify) → Phase D (audit + heal) → network.db
```

**Query time** (milliseconds, runs per question):
```
User question → LLM decomposition → navigate_models →
4 SQL queries → arithmetic scoring → ranked results
```

The entire build pipeline exists to produce `network.db`. At query time, no LLM runs inside ModelAtlas — the LLM is the *client*, not a dependency.

---

## Evidence: What's in the Database

```
models                 19,498 entries
bank_positions         ~80K  (8 banks × models with positions)
anchors                  166  labels across 8 banks
model_anchors         128K+  model-anchor links with confidence
model_links            ~2K   explicit relationships (fine_tuned_from, etc.)
model_metadata         ~60K  key-value pairs (summaries, scores, hashes)
```

The database is ~80MB. No external services required at query time. The entire semantic network is local.

---

## What This Is Not

- **Not a cloud service.** Everything runs on the user's machine. The database is a file.
- **Not an LLM wrapper.** No LLM runs at query time. The intelligence split is deliberate: LLM for decomposition, deterministic math for scoring.
- **Not a single-tool system.** Nine tools cover different interaction modes (navigate, compare, inspect, build). The LLM selects the right one.

---

## Related Concepts

- [Query Engine](Query-Engine) — how `navigate_models` scoring works internally
- [Data Model](Data-Model) — the database schema in detail
- [Extraction Pipeline](Extraction-Pipeline) — how build-time data gets in
- [Getting Started](Getting-Started) — how to set up the MCP client

---

*[← Navigation Geometry](Navigation-Geometry) · [Query Engine →](Query-Engine)*
