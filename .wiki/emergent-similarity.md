---
generated: true
generated_from: 
  - docs/wiki/emergent-similarity.md
source_hash: fdaf537c58e046ae
spec_hash: bfbb62b96f8e3def
file_hash: b7bc461199df01c9
materializer_version: "0.2.0"
theory_scope: false
audience: theory
page_id: emergent-similarity
---

# Emergent Similarity


# Emergent Similarity

**Two models sharing 15 [anchors](Glossary#anchor) are semantically related without an explicit edge between them. Similarity is not stored — it emerges from vocabulary overlap. Every similarity score traces back to specific shared labels. Nothing is an opaque embedding.**

---

## Problem Space

Most similarity systems use one of two approaches:

**Explicit edges.** Curate a knowledge graph with "similar_to" relationships. This requires human judgment for every pair, doesn't scale, and produces a graph that's always incomplete. With 19K models, you'd need to evaluate 180 million pairs.

**Embedding similarity.** Encode models as vectors, measure cosine distance. This scales, but the similarity is opaque — "these two models are 0.87 similar" gives no insight into *why*. You can't debug it, explain it, or trust it for novel queries.

Anchor-based similarity takes a third path: similarity that's both **scalable** and **interpretable**.

---

## Core Model

### The Anchor Dictionary

A shared vocabulary of ~170 semantic labels. Each anchor belongs to one [bank](Glossary#bank) and has a provenance (`bootstrap`, `deterministic`, `pattern`, `vibe`, `expansion`).

| Bank | Example Anchors |
|------|----------------|
| ARCHITECTURE | `transformer`, `mamba`, `mixture-of-experts`, `vision-transformer` |
| CAPABILITY | `instruction-following`, `tool-calling`, `code-generation`, `reasoning` |
| EFFICIENCY | `7B-class`, `sub-1B`, `quantized`, `consumer-GPU-viable` |
| COMPATIBILITY | `GGUF-available`, `MLX-compatible`, `Apple-Silicon-native` |
| LINEAGE | `Llama-family`, `Mistral-family`, `Qwen-family`, `fine-tune` |
| DOMAIN | `code-domain`, `medical-domain`, `legal-domain`, `math-domain` |
| QUALITY | `trending`, `high-downloads`, `community-favorite`, `high-mmlu` |
| TRAINING | `sft-trained`, `rlhf-trained`, `dpo-trained`, `distilled` |

Each model links to the anchors that describe it, with a confidence weight. The anchor dictionary grows organically — new labels are minted during extraction and expansion.

### How Similarity Emerges

Two models are similar when they share anchors. The more (and rarer) the shared anchors, the more similar.

**Example:** Qwen2.5-Coder-7B and CodeLlama-7B-Instruct might share:
- `decoder-only` (common — low signal)
- `code-generation` (moderately common)
- `instruction-following` (moderately common)
- `7B-class` (moderately common)
- `code-domain` (less common)

They differ on:
- `Qwen-family` vs `Llama-family`
- `dpo-trained` vs `sft-trained`

The shared anchors establish structural similarity; the differing anchors explain *how* they differ. No explicit "similar_to" edge was curated. No embedding was trained. The similarity is a consequence of how the models were described.

### IDF Weighting

Not all anchors carry equal signal. [IDF weighting](Glossary#idf-weighting) ensures that sharing a rare anchor matters more than sharing a common one:

```
IDF(anchor) = log(N / count_models_with_anchor)
```

| Anchor | Models with it | IDF weight |
|--------|---------------|------------|
| `decoder-only` | 17,000 | 0.14 |
| `instruction-following` | 4,200 | 1.53 |
| `code-generation` | 890 | 3.09 |
| `proof-assistant` | 12 | 7.39 |

Sharing `proof-assistant` is 50x more informative than sharing `decoder-only`. This falls directly out of information theory — rare events carry more information.

### Similarity as Set Operation

The query engine computes IDF-weighted Jaccard similarity:

```
similarity = sum(IDF[shared]) / sum(IDF[all unique])
```

This is a set intersection, not a dot product. It runs on anchor IDs, not vectors. It's exact, not approximate. And every term in the sum corresponds to a named anchor that can be inspected.

---

## Evidence: Why Not Embeddings

| Property | Embeddings | Anchor Overlap |
|----------|-----------|----------------|
| Interpretability | Opaque | Every component is a named label |
| Debuggability | "Similarity is 0.87" | "They share code-generation, 7B-class, instruction-following" |
| Query-time cost | Matrix multiply (GPU) | Set intersection (CPU) |
| Storage | 768+ floats per model | ~15 anchor IDs per model |
| New model | Requires re-encoding | Link to existing anchors |
| Rare concepts | Drowned by common features | IDF-amplified |

The anchor approach sacrifices the continuous granularity of embeddings for something more valuable in this context: **every similarity judgment is auditable**.

---

## What This Is Not

- **Not a knowledge graph.** There are no curated "similar_to" edges. Similarity is a computed consequence, not a stored fact.
- **Not a tagging system.** Tags are flat and unweighted. Anchors have bank affiliation, IDF weighting, confidence scores, and provenance tracking.
- **Not lossy compression.** The anchor set is the complete structural description. Nothing is projected away.

---

## Related Concepts

- [Anchor](Glossary#anchor) — what anchors are and how they're sourced
- [IDF weighting](Glossary#idf-weighting) — the information-theoretic foundation
- [Query Engine](Query-Engine) — how anchor overlap feeds into scoring
- [Signed Hierarchies](Signed-Hierarchies) — the other half (positions vs labels)

---

*[← Signed Hierarchies](Signed-Hierarchies) · [Navigation Geometry →](Navigation-Geometry)*
