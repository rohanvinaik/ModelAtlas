# Navigation Geometry

**Model discovery is not a search problem. It's a navigation problem. The difference: search finds things that match a description. Navigation finds things that are *in a direction* from where you are.**

---

## Problem Space

Every model search system — HuggingFace, Papers With Code, Ollama's library — answers the same type of question: "show me things that match these keywords/filters." This works when you know exactly what you want. It fails when you're exploring.

Exploration queries are directional:
- "Something like this but smaller"
- "More code-focused, less general"
- "What's between Llama-7B and GPT-4 in capability?"

These require a **coordinate system** — a space where "direction" and "distance" have meaning. Filter UIs can't express direction. Keyword search can't express proximity. You need geometry.

---

## Core Model

### Movement, Not Matching

A traditional query: `task=text-generation AND tags CONTAINS 'code' AND downloads > 1000`

This is a membership test. A model either passes all filters or doesn't. There's no gradient — no sense of "close but not quite." And there's no direction — no way to say "more of this, less of that."

A ModelAtlas query:

```
efficiency: -1   (move toward smaller)
capability: +1   (move toward more specialized)
require: [code-generation]
prefer: [tool-calling]
```

This describes a **trajectory** through model space. Every model in the database gets a score based on how well-aligned it is with this trajectory. Models near the target region score high. Models in the opposite direction score low. Models orthogonal to the query (right capability, wrong size) score in between.

### Multiplicative Scoring

The key design choice: bank alignment scores are **multiplied**, not averaged.

```
final_score = bank_alignment × anchor_relevance × seed_similarity
```

With averaging, a model that perfectly matches capability (+1.0) but completely fails efficiency (0.0) scores 0.5 — plausibly ranked. With multiplication, it scores 0.0 — correctly eliminated.

This reflects how humans evaluate models. "Perfect for code but way too large" is not a 50% match. It's a non-starter. Multiplicative scoring encodes this directly.

### The Zero Shortcut

Placing [zero states](Glossary#zero-state) at the query-frequency mode creates a computational shortcut: the most common type of query — "give me a good general model" — resolves entirely at the origin.

A query with no bank directions specified matches zero-position models with score 1.0. No computation needed beyond the default. Queries that specify directions pay proportional compute — one dimension of scoring per specified bank.

This isn't an optimization afterthought. It falls out of the zero placement design: the most popular query is the cheapest query because "what most people want" is literally the origin of the coordinate system.

---

## Evidence: Four Queries, Not 18K

The [query engine](Query-Engine) executes four SQL queries to score the entire model space:

1. **Pre-filter** — SQL `HAVING` on required [anchors](Glossary#anchor) eliminates non-candidates before scoring
2. **Batch positions** — one query loads all candidate [bank](Glossary#bank) positions
3. **Batch anchors** — one query loads all candidate anchor sets
4. **Batch authors** — one query loads display metadata

Scoring is arithmetic on in-memory dicts. The database does set operations (which models have these anchors?); Python does multiplication (how well does each position align?).

This replaces the naive approach of 18K individual `get_model()` calls with four indexed queries. The geometry makes this possible — batch position lookup works because every model's position is in the same coordinate space.

---

## What This Is Not

- **Not a vector store.** There are no embeddings, no approximate nearest neighbor search, no dimensional reduction. The space has exactly eight dimensions, all interpretable.
- **Not graph traversal.** While the network has explicit links, the primary query mechanism is coordinate-based scoring, not pathfinding. [Spreading activation](Glossary#spreading-activation) exists for "models like X" queries but is secondary to the structured engine.
- **Not a recommendation system.** There's no user modeling, no collaborative filtering, no "people who liked this also liked." Every score is computed from the model's structural properties relative to the query.

---

## Related Concepts

- [Signed Hierarchies](Signed-Hierarchies) — the coordinate system that makes navigation possible
- [Emergent Similarity](Emergent-Similarity) — the anchor layer that enriches navigation with semantic labels
- [Query Engine](Query-Engine) — the implementation: how coordinates and anchors become scores
- [Bank alignment](Glossary#bank-alignment) — the specific scoring function

---

*[← Emergent Similarity](Emergent-Similarity) · [System Overview →](System-Overview)*
