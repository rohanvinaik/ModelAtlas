# hf-model-search — System Design Document

This document is the complete architectural specification. A new agent should be able to recover the entire design from this file alone.

## What This Is

A **hierarchical semantic network** of ML models, exposed as an MCP server. Models are positioned across multiple orthogonal **semantic banks** via signed distances from meaningful zero states, and connected through a shared **anchor dictionary** that creates emergent similarity. A flat SQL table exists as a sidecar for overflow metadata that doesn't decompose cleanly into the network.

The query model is **navigational**, not filtering. You don't ask "give me rows matching these WHERE clauses." You ask "what's near this point in model space?", "navigate toward more general in CAPABILITY while staying small in EFFICIENCY", "what exists that I could fine-tune for this task?"

## What This Is NOT

- NOT a vector store / embedding search engine
- NOT a flat SQL database with 65 columns and WHERE clauses
- NOT a thin wrapper around HuggingFace's API
- NOT a system that stores or embeds raw model card text

The value: HF exposes ~10 structured fields. This tool extracts/derives relational and hierarchical information about models — their capabilities, lineage, efficiency profiles, architectural properties, domain specializations — and encodes them as **positions in a structured semantic space**. This enables compound navigational queries HF can't support: "most general base model in the Llama family that supports tool-calling and runs on consumer GPU" or "models architecturally similar to Mamba but with instruction tuning."

## Core Architecture: The Semantic Network

Inspired by the Sparse Wiki Grounding architecture (see `/Users/rohanvinaik/sparse-wiki-grounding/reports/ARCHITECTURE_DEEP_DIVE.md` for the theoretical foundation). The key ideas borrowed:

1. **Signed hierarchies with zero states** — every model has a signed distance from a semantically meaningful "zero" in each bank. Zero is placed at the query-frequency mode (the most commonly searched region), so most queries resolve near the origin.

2. **Orthogonal semantic banks** — multiple independent dimensions, each tracking a different aspect of model identity. Banks prevent semantic bleeding: searching for "good at code" activates CAPABILITY, not EFFICIENCY.

3. **Anchor dictionary** — shared vocabulary of characteristics (like "instruction-following", "RLHF-tuned", "Apple-Silicon-native") that models link to. Models sharing anchors are semantically related without needing explicit edges. Anchor sets enable set-operation queries (intersection = shared capabilities, symmetric difference = distinguishing features, Jaccard = overall similarity).

4. **Two-layer connectivity** — Layer 1: explicit model relationships (fine-tuned-from, same-family, quantized-from). Layer 2: implicit similarity through shared anchors.

5. **Auditable, not black-box** — every similarity score traces back to specific shared anchors and bank positions. No opaque embeddings.

### The Seven Semantic Banks

Each bank has a zero state (the semantic origin), with signed distances indicating direction from zero. Negative = toward abstract/general/earlier. Positive = toward specific/specialized/derived.

```
ARCHITECTURE    Zero: standard transformer decoder (the mode — most models, most queries)
                -N: simpler/older (encoder-only, encoder-decoder, RNN-era)
                +N: novel/specialized (Mamba, RWKV, SSM, hybrid, mixture-of-experts)
                WHY ITS OWN BANK: cleanest, most numerical signal. ~95% of model
                filtering can be executed by querying this bank alone.

CAPABILITY      Zero: general language model
                -N: narrower capability (single-task classifiers, embedding-only)
                +N: richer capability (code, reasoning, creative writing, tool-calling,
                     NER, orchestration, structured output, function calling)
                ANCHORS: "instruction-following", "tool-calling", "code-generation",
                         "creative-writing", "NER", "orchestration", "time-series"

EFFICIENCY      Zero: ~7B mainstream (the sweet spot most people search for)
                -N: smaller/lighter (3B, 1B, 0.5B, micro/embedded)
                +N: larger/heavier (13B, 30B, 70B, frontier)
                DERIVED FROM: parameter_count, quantization, memory_footprint

COMPATIBILITY   Zero: standard transformers + PyTorch (universal baseline)
                +N: more specific format/framework/hardware target
                    +1: specific framework (MLX, llama.cpp, vLLM, TensorRT)
                    +2: specific format (GGUF, GPTQ, AWQ, EXL2, safetensors)
                    +3: specific hardware optimization (Apple Silicon, specific GPU arch)

LINEAGE         Zero: base/foundational model of a family
                -N: EARLIER architectures in the same family (GPT-2 is negative
                    relative to GPT-4 — they share a family even though one isn't
                    fine-tuned from the other. Predecessors go negative.)
                +N: DERIVED models (fine-tune → quantized → community derivative)
                    +1: official variant (size variant, instruct version)
                    +2: fine-tune (LoRA, DPO, RLHF, domain adaptation)
                    +3: community derivative (merged, quantized, distilled)

DOMAIN          Zero: general knowledge (broad training data, no specialization)
                +N: increasingly narrow domain specialization
                    +1: broad domain (code, science, legal, medical, finance)
                    +2: narrow domain (Python, constitutional law, radiology)
                    +3: ultra-narrow (Apple Shortcuts DSL, SEC filings, specific game lore)

QUALITY         Zero: established, well-known, mainstream adoption
                -N: legacy, abandoned, superseded, low community engagement
                +N: trending, high momentum, rising community adoption
                DERIVED FROM: likes, downloads, download_velocity, days_since_release
```

### The Anchor Dictionary

A shared vocabulary of semantic labels that models link to. Each anchor belongs to a bank (routes activation through that bank's channel). Anchors create **emergent connections** — models sharing anchors cluster together without explicit edges.

**Bootstrap strategy**: Start from known tags, capabilities, format names extracted from HF metadata and model cards. Grow organically as more models are indexed. No need for a perfect upfront vocabulary — vibe-y bootstrapping, refined over time.

**Example anchors** (bank assignment in parentheses):
- "instruction-following" (CAPABILITY)
- "RLHF-tuned" (CAPABILITY)
- "tool-calling" (CAPABILITY)
- "code-generation" (CAPABILITY)
- "orchestration" (CAPABILITY)
- "consumer-GPU-viable" (EFFICIENCY)
- "Apple-Silicon-native" (COMPATIBILITY)
- "GGUF-available" (COMPATIBILITY)
- "MLX-compatible" (COMPATIBILITY)
- "Llama-family" (LINEAGE)
- "Mistral-family" (LINEAGE)
- "legal-domain" (DOMAIN)
- "transformer" (ARCHITECTURE)
- "mixture-of-experts" (ARCHITECTURE)
- "trending" (QUALITY)

### Database Schema

```sql
-- Models: the entities
models (
    model_id    TEXT PRIMARY KEY,  -- e.g. "meta-llama/Llama-3.1-8B-Instruct"
    author      TEXT,
    source      TEXT DEFAULT 'huggingface',  -- huggingface, ollama, replicate, civitai
    display_name TEXT
);

-- Bank positions: signed positions across the 7 banks
model_positions (
    model_id    TEXT REFERENCES models,
    bank        TEXT,          -- ARCHITECTURE, CAPABILITY, EFFICIENCY, etc.
    path_sign   INTEGER,       -- -1 or +1
    path_depth  INTEGER,       -- distance from zero
    path_nodes  TEXT,          -- JSON: path from zero to this position
    zero_state  TEXT,          -- the zero reference label for this bank
    PRIMARY KEY (model_id, bank)
);

-- Model links: explicit relationships (Layer 1)
model_links (
    source_id   TEXT REFERENCES models,
    target_id   TEXT REFERENCES models,
    relation    TEXT,          -- 'fine_tuned_from', 'quantized_from', 'same_family',
                               -- 'predecessor', 'successor', 'variant_of'
    weight      REAL DEFAULT 1.0
);

-- Anchor dictionary: shared capability vocabulary
anchors (
    anchor_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    label       TEXT UNIQUE,   -- 'instruction-following', 'RLHF-tuned', etc.
    bank        TEXT,          -- which bank this anchor activates
    category    TEXT           -- optional grouping
);

-- Model-anchor links: which models have which characteristics (Layer 2)
model_anchors (
    model_id    TEXT REFERENCES models,
    anchor_id   INTEGER REFERENCES anchors,
    weight      REAL DEFAULT 1.0,
    PRIMARY KEY (model_id, anchor_id)
);

-- Overflow metadata: flat fields that don't decompose into the network
-- In a perfect design this table would be empty. In practice it catches
-- data that's just a number or string with no meaningful hierarchy.
model_metadata (
    model_id    TEXT REFERENCES models,
    key         TEXT,          -- 'sha', 'created_at', 'gated', 'context_length',
                               -- 'parameter_count', 'license', benchmark scores, etc.
    value       TEXT,
    value_type  TEXT,          -- 'int', 'float', 'str', 'bool', 'datetime'
    PRIMARY KEY (model_id, key)
);

-- Key indices (bidirectional anchor lookup, as in sparse-wiki)
CREATE INDEX idx_positions_bank ON model_positions(bank);
CREATE INDEX idx_links_source ON model_links(source_id);
CREATE INDEX idx_links_target ON model_links(target_id);
CREATE INDEX idx_model_anchors_model ON model_anchors(model_id);
CREATE INDEX idx_model_anchors_anchor ON model_anchors(anchor_id);  -- "what models share this anchor?"
CREATE INDEX idx_metadata_key ON model_metadata(key);
```

### Query Model

Queries are **navigational**, not just filtering. Examples:

| Query | How It Works |
|-------|-------------|
| "Models like X" | Jaccard similarity on anchor sets, weighted by bank |
| "Most general base for fine-tuning on code" | Find models with "code-generation" anchor, sort by lowest path_depth in LINEAGE (closest to zero = most general base) |
| "Small model with tool-calling" | EFFICIENCY bank position < 0 (small) + CAPABILITY anchor "tool-calling" |
| "What distinguishes Model A from Model B?" | Symmetric difference of anchor sets |
| "Navigate from X toward smaller and more code-focused" | Decrease EFFICIENCY position, increase CAPABILITY toward code anchors |
| "Llama family tree" | Traverse LINEAGE bank: all models with "Llama-family" anchor, ordered by signed position |
| "Trending models in the Mamba architecture space" | ARCHITECTURE bank position > 0 (novel) + QUALITY bank position > 0 (trending) |

**Set operations on anchor sets** (from sparse-wiki):
- **Intersection** (A & B) = shared capabilities between two models
- **Symmetric difference** (A ^ B) = what makes them different
- **Jaccard similarity** (|A & B| / |A | B|) = overall semantic overlap

### The One Natural Language Field

`vibe_summary`: one sentence per model capturing the irreducible "feel" — what it's known for, what makes it distinctive. This is the ONLY prose stored. Everything else is numbers, categories, positions, and anchors.

This field lives in the overflow metadata table (or optionally in a small ChromaDB sidecar for semantic search over vibes). It handles the queries that no structured decomposition can fully capture.

## Source Adapters

Pluggable fetchers that all produce the same output format: model entities with bank positions, anchor links, metadata, and explicit relationships. Each source has different raw data but maps to the same schema.

| Source | What It Provides | Adapter Status |
|--------|-----------------|---------------|
| HuggingFace | Richest metadata — API fields + model cards + configs + file lists | Primary, build first |
| Ollama | Direct hardware compatibility info, simpler metadata | Future |
| Replicate | API-oriented models, pricing info, latency data | Future |
| CivitAI | Image/diffusion models, community ratings | Future |

Every indexed model carries a `source` field. The network is source-agnostic — a Llama model from HF and the same model from Ollama occupy the same point in semantic space and link to the same anchors.

## Extraction Pipeline

The hard part. Reads model cards, configs, API metadata, and file lists. Produces: bank positions, anchor links, explicit model relationships, and overflow metadata.

**Three tiers of extraction reliability:**

1. **Deterministic** (from API / config.json / safetensors metadata):
   Parameter count, context length, architecture type, vocab size, embedding dim, num layers, num heads, file formats available, author, license, dates, download counts.

2. **Pattern matching** (from tags / file names / model card structure):
   Is instruction-tuned, is chat model, is code model, quantization formats, base model reference, fine-tune method, has chat template, supported languages, benchmark scores (often in tables in model cards).

3. **NLP/LLM extraction** (from model card prose — the hard part):
   Training data domains, capability signals, the vibe_summary sentence. This is where a small local embedding model or LLM call fills gaps that symbolic parsing can't handle. "Vibe coding" — a mish-mash of regex, heuristics, and LLM calls papered over to get 99% of what a careful human would extract.

## Update Strategy

Model data is **mostly static**. A model's parameter count, architecture, and training data don't change after release. The real "update" operation is **discovering and adding new models**, not updating existing entries.

- **Periodic additive sweeps**: Find new models above quality thresholds, run through extraction pipeline, INSERT into network. Existing nodes rarely need touching.
- **Download/like snapshots**: Capture periodically to compute velocity (QUALITY bank signal). Store snapshots in metadata.
- **Lineage updates**: When a new fine-tune appears, add it to the network with a `fine_tuned_from` link to its base. The LINEAGE bank position is computed from the link structure.

## Current State of Code

- MCP server skeleton exists and works (5 tools registered, server connects via FastMCP)
- Layer 1 (HF API structured search) and Layer 2 (RapidFuzz fuzzy matching) are reusable
- Layer 3 (ChromaDB semantic search) was built around the wrong architecture and needs complete replacement with the network-based approach
- Dependencies installed: mcp, huggingface-hub, sentence-transformers, chromadb, rapidfuzz
- The network database (SQLite with the schema above) needs to be built from scratch
- The extraction pipeline needs to be built from scratch
- All documentation has been updated to reflect the new architecture

## Guardrails

1. **Do not embed raw model card text.** Model cards are an INPUT SOURCE for the extraction pipeline. They are read, analyzed, decomposed into structured positions/anchors/metadata, then discarded.
2. **Do not store prose** except the one `vibe_summary` field per model.
3. **Do not treat this as a filtering problem.** Queries are navigational — exploring a semantic space, not running WHERE clauses.
4. **The network is the primary storage, SQL is overflow.** If you're putting most data in the metadata table, you're doing it wrong.
5. **Signed hierarchies apply across ALL banks**, not just lineage. Every bank has a zero state and models have signed distances from it.
6. **Anchors create emergent similarity.** Two models sharing 15 anchors are similar even without an explicit edge. The anchor dictionary IS the semantic vocabulary.
7. **Source-agnostic.** The same model from different sources occupies the same point in semantic space.
8. **Talk before building.** The architecture IS the project. Don't start coding without understanding this document.
