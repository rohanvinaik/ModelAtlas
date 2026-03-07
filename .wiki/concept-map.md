---
generated: true
generated_from: 
  - docs/wiki/concept-map.md
source_hash: bc5fd2de19780a21
spec_hash: 00dabc9cc59164af
file_hash: c0527ab5460c9cb0
materializer_version: "0.2.0"
theory_scope: false
audience: user
page_id: concept-map
---

# Concept Map


# Concept Map

**How the core ideas connect — from the problem through the theory to the engineered solution.**

---

## The Conceptual Flow

```
THE PROBLEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  HuggingFace has 1M models and ~10 structured fields.
  Capabilities, lineage, efficiency — trapped in prose.
  Search finds keywords. Discovery needs navigation.

                        ↓
                   The Gap
                        ↓

THE THEORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─────────────────────┐     ┌────────────────────────┐
  │  Signed Hierarchies │     │  Emergent Similarity   │
  │                     │     │                        │
  │  8 banks with zero  │     │  Anchor labels shared  │
  │  states and signed  │     │  across models.        │
  │  positions give     │     │  Overlap = similarity. │
  │  direction, not     │     │  IDF weighting makes   │
  │  categories.        │     │  rare anchors matter.  │
  └────────┬────────────┘     └───────────┬────────────┘
           │                              │
           └──────────┬───────────────────┘
                      ↓
            Navigation Geometry

            Queries are movements through
            space, not WHERE clauses.
            Multiplicative scoring across
            banks. Four SQL queries, not 18K.

                      ↓

THE SYSTEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │ Extraction   │  │ Data Model   │  │ Query Engine │
  │ Pipeline     │  │              │  │              │
  │              │  │ 6 tables,    │  │ navigate_    │
  │ A: fetch     │  │ 8 banks,     │  │ models:      │
  │ B: extract   │→ │ anchors,     │→ │ bank align × │
  │ C: classify  │  │ links        │  │ anchor rel × │
  │ D: audit     │  │              │  │ seed sim     │
  └──────────────┘  └──────────────┘  └──────┬───────┘
                                             ↓
                                    ┌──────────────┐
                                    │ MCP Server   │
                                    │ 9 tools      │
                                    │ SQLite local  │
                                    └──────┬───────┘
                                           ↓
                                    LLM decomposes
                                    the question.
                                    ModelAtlas does
                                    the math.
```

---

## Reading Order by Interest

### "I just want to use it"
[Getting Started](Getting-Started) → [Query Examples](Query-Examples)

### "I want the full theory-to-implementation story"
[The Gap](The-Gap) → [Signed Hierarchies](Signed-Hierarchies) → [Emergent Similarity](Emergent-Similarity) → [Navigation Geometry](Navigation-Geometry) → [Query Engine](Query-Engine) → [Data Model](Data-Model)

### "I want to understand the architecture"
[System Overview](System-Overview) → [Query Engine](Query-Engine) → [Data Model](Data-Model) → [Extraction Pipeline](Extraction-Pipeline) → [Data Distribution](Data-Distribution)

### "I want one concept explained well"
Every page is self-contained with a promise, problem space, core model, evidence, and fencing. Start anywhere the [Glossary](Glossary) points you.

---

## Concept Dependencies

| Concept | Depends On | Enables |
|---------|-----------|---------|
| [The Gap](The-Gap) | (none) | Everything — defines the problem |
| [Signed Hierarchies](Signed-Hierarchies) | The Gap | Navigation Geometry, Query Engine |
| [Emergent Similarity](Emergent-Similarity) | Signed Hierarchies | Navigation Geometry, Query Engine |
| [Navigation Geometry](Navigation-Geometry) | Signed Hierarchies, Emergent Similarity | Query Engine |
| [Query Engine](Query-Engine) | Navigation Geometry | (application layer) |
| [Data Model](Data-Model) | Signed Hierarchies, Emergent Similarity | Query Engine, Extraction Pipeline |
| [Extraction Pipeline](Extraction-Pipeline) | Data Model | Data Distribution |
| [Data Distribution](Data-Distribution) | (none) | Getting Started |

---

*[← Glossary](Glossary) · [Home](Home)*
