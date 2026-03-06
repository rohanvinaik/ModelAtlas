# ModelAtlas

**By reading this wiki, you will understand why model discovery is a navigation problem — and how ModelAtlas solves it with signed hierarchies and anchor-based similarity. No embeddings, no GPU at query time. Just integers and set intersections.**

---

## What ModelAtlas Does

HuggingFace Hub has a million models and roughly ten structured fields per model. The relational information that makes models *findable* — capabilities, lineage, efficiency profiles, architectural properties, domain specializations — is trapped in unstructured model cards, naming conventions, config files, and community knowledge.

ModelAtlas extracts that information and encodes it as **positions in a structured semantic space**. The result is a navigable network where queries are movements through space ("toward smaller and more code-focused"), not WHERE clauses ("parameter_count < 3B AND tags LIKE '%code%'").

The entire system is a SQLite file, a vocabulary of ~170 [anchor](Glossary#anchor) labels, and signed integers. No GPU at query time. No vector store in the background. No running services.

---

## Choose Your Path

### I want to use it
Start with **[Getting Started](Getting-Started)** for installation and your first query, then browse **[Query Examples](Query-Examples)** for the kinds of questions ModelAtlas can answer that HuggingFace cannot.

### I want to understand how it works
**[System Overview](System-Overview)** gives the 30-second architecture map. From there, dive into the **[Query Engine](Query-Engine)** (how scoring works), the **[Extraction Pipeline](Extraction-Pipeline)** (how data gets in), or the **[Data Model](Data-Model)** (what the database looks like).

### I want to understand why it's designed this way
**[The Gap](The-Gap)** defines the problem: what's missing from HuggingFace and why it matters. **[Signed Hierarchies](Signed-Hierarchies)** introduces the core model. **[Emergent Similarity](Emergent-Similarity)** explains why anchor overlap replaces embeddings. **[Navigation Geometry](Navigation-Geometry)** makes the case for movement through space over filter queries.

---

## The Concept in One Diagram

```
User asks: "small code model with tool-calling that runs on a Mac"

LLM decomposes into:
  EFFICIENCY  → negative (small)
  CAPABILITY  → positive (code, tool-calling)
  COMPATIBILITY → positive (MLX/Apple Silicon)
  require: [code-generation]
  prefer:  [tool-calling, Apple-Silicon-native]

ModelAtlas does:
  1. SQL pre-filter on required anchors
  2. Bank alignment scoring (signed distance from zero)
  3. IDF-weighted anchor overlap
  4. Multiplicative combination → ranked results

No embeddings computed. No model loaded. Just arithmetic on integers
and set intersections on small lists.
```

The intelligence is in the interaction — the LLM decomposes the question, ModelAtlas does the math. Neither is smart alone, but the system is.

---

## Current State

ModelAtlas is in active beta. The network contains **19,498 models** with **166 anchors** across 8 [banks](Glossary#bank), **128K+ model-anchor links**, and a multi-phase correction pipeline actively improving accuracy. See [Data Distribution](Data-Distribution) for how to get the pre-built network.

---

*[Glossary](Glossary) · [Getting Started →](Getting-Started)*
