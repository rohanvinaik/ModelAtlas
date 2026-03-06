---
generated: true
generated_from: 
  - docs/wiki/signed-hierarchies.md
source_hash: 91f7e2abc335ef35
spec_hash: 2afc1aa246bc1dfa
file_hash: 827dae34472f3750
materializer_version: "0.2.0"
theory_scope: false
audience: theory
page_id: signed-hierarchies
---

# Signed Hierarchies


# Signed Hierarchies

**The core model: every ML model has a position along eight independent dimensions, with signed distance from a zero point placed at what most people are looking for. "Small" is not a category — it's a direction.**

---

## Problem Space

The natural instinct for organizing models is categories: size buckets (small/medium/large), task types (chat/code/medical), architecture labels (transformer/SSM). But categories fail at the exact moment you need them most — when a query spans multiple dimensions.

"Small code model with tool-calling" requires reasoning about efficiency (how small?), capability (code + tool-calling), and the *relationship* between them (small relative to what?). Categorical systems can't express proximity. A 3B model is *close* to 7B — not a binary mismatch with "medium."

Signed hierarchies solve this by replacing categories with **positions on directed axes**.

---

## Core Model

### The Eight Banks

Eight orthogonal dimensions. Each has a [zero state](Glossary#zero-state) placed at the query-frequency mode — the thing most people are looking for.

```
Bank            Zero State                  ← Negative              Positive →
─────────────── ─────────────────────────── ───────────────────── ──────────────────────
ARCHITECTURE    transformer decoder-only    encoder-only, RNN     Mamba, RWKV, MoE
CAPABILITY      general language model      single-task, embed    code, tools, reasoning
EFFICIENCY      ~7B parameters              sub-1B, 1B, 3B       13B, 30B, 70B, frontier
COMPATIBILITY   PyTorch + transformers      —                     GGUF, MLX, Apple Silicon
LINEAGE         base/foundational model     predecessors          fine-tune, quant, merge
DOMAIN          general knowledge           —                     code, medical, legal
QUALITY         established mainstream      legacy, abandoned     trending, rising
TRAINING        standard supervised (SFT)   LoRA, distillation    RLHF, DPO, multi-stage
```

### Signed Position

A model's position in each bank is stored as `(sign, depth)`:
- `(0, 0)` — at zero. The mainstream case.
- `(-1, 2)` — two steps negative. Smaller, simpler, earlier.
- `(+1, 3)` — three steps positive. Larger, more specialized, more derived.

The signed position is `sign × depth`. This gives a single integer that supports arithmetic: distance between two models is the absolute difference of their signed positions.

### Why Signed, Not Categorical

A categorical "size" field with values `{small, medium, large}` can't express:
- **Proximity.** Is "small" closer to "medium" or to "tiny"? Categories don't know.
- **Direction.** "Toward smaller" is a meaningful query. "Toward small" is a value match.
- **Gradient scoring.** With signed positions, a 3B model scores 0.8 against a "small" query and 0.3 against a "large" query. With categories, it's 1.0 or 0.0.

### Why Zero at the Mode

Zero is placed at **what most people look for**, not at a mathematical center. For EFFICIENCY, zero is ~7B because that's the mainstream sweet spot — enough capability for most tasks, runnable on consumer hardware.

This has a computational consequence: most queries resolve near the origin. A query with no efficiency preference matches zero-position models perfectly (score = 1.0) and penalizes extremes. The common case is the fast case.

---

## Evidence: Scoring in Action

A query for "small code model":
- EFFICIENCY direction = -1 (small)
- CAPABILITY direction = +1 (specialized)

| Model | EFFICIENCY pos | CAPABILITY pos | Bank score |
|-------|---------------|----------------|------------|
| Qwen2.5-Coder-1.5B | (-1, 2) | (+1, 2) | 1.0 × 1.0 = 1.0 |
| CodeLlama-7B | (0, 0) | (+1, 2) | 0.5 × 1.0 = 0.5 |
| CodeLlama-34B | (+1, 2) | (+1, 2) | 0.33 × 1.0 = 0.33 |
| Llama-3.1-8B | (0, 0) | (0, 0) | 0.5 × 0.5 = 0.25 |

The 1.5B coder scores highest because it's aligned in *both* dimensions. The 34B coder scores lower — right capability, wrong size. The general 8B model scores lowest — neither aligned. The scores are multiplicative: excellence in one dimension doesn't compensate for misalignment in another.

---

## What This Is Not

- **Not embeddings.** There are no learned vectors. Positions are assigned by deterministic extraction and LLM classification, then audited.
- **Not a ranking system.** There's no "best model" score. Just "what's near here, and what path leads where you need."
- **Not fixed categories.** The depth within each bank can grow as new models arrive. The structure is extensible without schema changes.

---

## Related Concepts

- [Zero state](Glossary#zero-state) — how zeros are chosen
- [Bank alignment](Glossary#bank-alignment) — how positions become scores
- [Emergent Similarity](Emergent-Similarity) — the other half of the model (anchors)
- [Navigation Geometry](Navigation-Geometry) — why this is navigation, not search

---

*[← The Gap](The-Gap) · [Emergent Similarity →](Emergent-Similarity)*
