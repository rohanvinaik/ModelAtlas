---
generated: true
generated_from: 
  - docs/wiki/query-examples.md
source_hash: fffbe0b5f019962c
spec_hash: e0a1da3350ce52a1
file_hash: ea59ecce40faeb8e
materializer_version: "0.2.0"
theory_scope: false
audience: user
page_id: query-examples
---

# Query Examples


# Query Examples

**A cookbook of questions ModelAtlas can answer that HuggingFace cannot — and the structural reasons why.**

---

## Problem Space

Search engines match keywords. Filter UIs match columns. Neither can express: "models like *this one* but smaller and more code-focused." That query requires understanding proximity in multiple dimensions simultaneously — a navigation problem, not a search problem.

Every example below shows a question, how the LLM decomposes it into [navigate_models](Glossary#navigate_models) parameters, and what structural features produce the answer.

---

## Navigation Queries

### "Small code model with tool-calling for a MacBook"

```
efficiency:    -1  (smaller than mainstream)
capability:    +1  (specialized)
compatibility: +1  (platform-specific)
require:  [code-generation]
prefer:   [tool-calling, Apple-Silicon-native, instruction-following]
```

**Why HF can't answer this:** No field combines size direction, capability specialization, and platform compatibility. You'd need to manually intersect task filters, guess at parameter counts, and grep model cards for "Apple Silicon."

**What ModelAtlas does:** [Bank alignment](Glossary#bank-alignment) scores penalize models on the wrong side of any dimension. The `require` filter uses SQL pre-filtering. [IDF weighting](Glossary#idf-weighting) ensures rare anchors like `Apple-Silicon-native` count more than ubiquitous ones like `decoder-only`.

### "Models architecturally similar to Mamba but instruction-tuned"

```
architecture:  +1  (novel architectures)
require:  [instruction-following]
prefer:   [mamba, ssm]
```

**Why HF can't answer this:** HuggingFace has no "architecture" axis. You can filter by `model_type`, but that's a flat string — `mamba` and `rwkv` don't know they're related (both are SSMs, both are non-transformer).

**What ModelAtlas does:** The ARCHITECTURE [bank](Glossary#bank) places Mamba at positive signed position (novel relative to the transformer [zero state](Glossary#zero-state)). SSM models cluster nearby. The `require` filter ensures only instruction-tuned variants return.

### "What's the base model for this GGUF quantization?"

```
similar_to: "TheBloke/Llama-3.1-8B-Instruct-GGUF"
lineage:    -1  (toward base/foundational)
```

**Why HF can't answer this:** GGUF files are uploaded by quantizers, not the original authors. The provenance chain (`quantized_from → fine_tuned_from → base model`) is implicit in naming conventions and model card text, not in any queryable field.

**What ModelAtlas does:** Explicit `quantized_from` and `fine_tuned_from` links in the network. The LINEAGE bank encodes derivation depth. Seeding from the GGUF model and moving negative in LINEAGE traverses back toward the base.

### "Trending medical models with strong benchmarks"

```
domain:   +1  (specialized)
quality:  +1  (above mainstream)
require:  [medical-domain]
prefer:   [trending, high-mmlu, instruction-following]
```

**Why HF can't answer this:** "Trending" is a sort order on HF, not a filterable property. "Medical" is a tag some authors apply. "Strong benchmarks" requires parsing model cards or external leaderboards.

**What ModelAtlas does:** The QUALITY bank tracks trend signals. The DOMAIN bank encodes specialization depth. Benchmark-derived anchors like `high-mmlu` are extracted during classification and linked with provenance.

---

## Comparison Queries

### "How do Llama 3.1 8B and Qwen 2.5 7B differ?"

Uses `compare_models` — set operations on [anchor](Glossary#anchor) sets plus per-[bank](Glossary#bank) position deltas.

Returns:
- **Shared anchors:** What they have in common (decoder-only, instruction-following, 7B-class)
- **Unique to each:** What distinguishes them (family, training approach, specific capabilities)
- **Bank deltas:** Where they sit differently across dimensions
- **Jaccard similarity:** Quantified structural overlap

### "What models are most similar to CodeLlama-34B?"

Uses `similar_to` — IDF-weighted Jaccard between the seed model's anchor set and every other model's. Returns the structurally nearest neighbors with the specific anchors that drove the similarity score.

---

## Discovery Queries

### "What kinds of models exist in the speech domain?"

```
domain: +1
prefer: [speech-domain, audio-domain]
```

Returns models clustered by speech-related anchors. The [anchor](Glossary#anchor) sets on the results reveal the sub-structure: text-to-speech, speech-recognition, voice-cloning, audio-generation — categories that emerge from the data rather than being imposed by a taxonomy.

### "Show me the frontier of efficient reasoning models"

```
efficiency: -1  (small)
capability: +1  (specialized)
training:   +1  (advanced training)
prefer: [reasoning, chain-of-thought, distilled]
```

The combination of opposing directions (small BUT capable, advanced training) naturally surfaces models that punch above their weight class — the interesting part of the design space.

---

## What These Examples Have in Common

Every query above requires **simultaneous reasoning across multiple dimensions** — something keyword search cannot do and filter UIs cannot express. The [signed hierarchy](Signed-Hierarchies) model converts each dimension into arithmetic, and the multiplicative scoring ensures no dimension is ignored.

---

## Related Concepts

- [Query Engine](Query-Engine) — how the scoring works internally
- [Signed Hierarchies](Signed-Hierarchies) — why directions and distances, not categories
- [Glossary](Glossary) — definitions of all terms used above

---

*[← Getting Started](Getting-Started) · [System Overview →](System-Overview)*
