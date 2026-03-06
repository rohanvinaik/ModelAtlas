---
generated: true
generated_from: 
  - docs/DESIGN.md
  - README.md
source_hash: 66420f957e540e9c
spec_hash: a7a7511f0bf718c9
file_hash: 04e1458709eded9f
materializer_version: "0.1.0"
theory_scope: false
audience: theory
page_id: theory-overview
---

# Theory: Signed Hierarchies & Semantic Navigation


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

### 2.6 Anchor Lifecycle

Anchors progress through a lifecycle of increasing confidence:

```
Bootstrap (seed dictionary, ~130 labels)
    │
    ▼
Extract (Tier 1 deterministic + Tier 2 pattern, confidence 0.8-1.0)
    │
    ▼
C2 Classify (3B LLM selects from dictionary, confidence 0.5)
    │
    ▼
D1 Audit (deterministic re-check, produces audit_score per model)
    │
    ▼
D2 Expand (add missing labels via strict DSL, confidence 0.7)
    │
    ▼
D3 Heal (LLM correction from raw evidence, confidence 0.6)
    │
    ▼
D4 Train (corrections → DPO JSONL for future C2 improvement)
```

Each stage has explicit provenance tracking:
- `anchors.source`: `bootstrap`, `deterministic`, `pattern`, `vibe`, `expansion`
- `model_anchors.confidence`: decreases with extraction tier uncertainty
- `correction_events`: full audit trail of original → healed with rationale
- `phase_d_runs`: every D-phase operation has a UUID, config, and summary

The provenance layer (`phase_d_runs`, `audit_findings`, `correction_events`) makes every classification decision auditable and every correction a training example.


## The gap

HuggingFace knows that `meta-llama/Llama-3.1-8B-Instruct` has 42,000 likes and uses the `transformers` library. What it doesn't know: this model is an instruction-tuned derivative of a base model in the Llama family, supports tool-calling, sits in the mainstream efficiency range, and has 47 quantized variants on the Hub. That information exists — scattered across model cards, naming conventions, config files, and community knowledge. But it's not queryable.

There isn't an API call or a search bar that answers:

- "What's the most general Llama base that supports tool-calling and fits on consumer GPU?"
- "What are some models that are architecturally similar to Mamba, but with instruction tuning?"
- "Can we find models like *this* one, but smaller and more code-focused?"

These aren't filter queries. They're **navigation** — and HuggingFace doesn't have a coordinate system to navigate with.


## The idea

Every model has a position along eight independent dimensions, with signed hierarchical traversal from an assigned zero point. Take efficiency: 7B is a mainstream sweet spot, so 7B is set as "zero". Smaller goes negative. Larger goes positive. "Small" just means "negative in EFFICIENCY."

```
ARCHITECTURE    zero = transformer decoder       →  +novel (Mamba, MoE)
CAPABILITY      zero = general language model     →  +rich (code, tools, reasoning)
EFFICIENCY      zero = ~7B parameters             →  +larger  / -smaller
COMPATIBILITY   zero = PyTorch + transformers     →  +specific (GGUF, MLX)
LINEAGE         zero = base/foundational model    →  +derived (fine-tune, quant)
DOMAIN          zero = general knowledge           →  +specialized (code, medical)
QUALITY         zero = established mainstream      →  +trending  / -legacy
TRAINING        zero = standard supervised (SFT)  →  +complex (RLHF, DPO) / -simpler (LoRA, distill)
```

Zero is defined as **the most common thing people look for**. Most queries resolve near the origin.

On top of coordinates, models share **anchors** — a vocabulary of characteristics like "instruction-following", "GGUF-available", "Llama-family." Models sharing anchors are similar without explicit edges. Similarity is emergent, and every score traces back to specific shared labels. Nothing is an opaque embedding.

The LLM decomposes a user's question into coordinates and anchors. ModelAtlas does arithmetic on integers and set intersections on small lists. The intelligence is in the interaction — no single piece is smart, but the system is.


## What this is not

- **Not a vector store.** No embeddings. Similarity comes from shared structure.
- **Not a database with 65 columns.** Eight signed dimensions and a label vocabulary replace flat attributes.
- **Not a HuggingFace wrapper.** HF is a data source. The value is the extracted structure HF doesn't expose.
- **Not a ranking system.** No "best model" score. Just "what's near here, and what path leads where you need."

The entire thing is a SQLite file, a few thousand anchor labels, and signed integers. No GPU at query time. No vector store in the background. No running services. Full semantic decomposition was done at home with spare compute. At query time, it's simply multiplication and set intersection.
