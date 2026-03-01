# ModelAtlas

You want to find "a small code model with tool-calling that runs on Apple Silicon." HuggingFace gives you a search bar and 10 filterable columns. You get 50,000 results, or zero, and no way to navigate between those extremes.

ModelAtlas fixes this. It's an MCP server that positions ML models in a **navigable semantic space** — so an LLM can explore what exists, not just filter what matches.

## The Problem

HuggingFace knows that `meta-llama/Llama-3.1-8B-Instruct` has 42,000 likes and uses the `transformers` library. It does not know that this model is an instruction-tuned derivative of a base model in the Llama family, that it supports tool-calling, that it's in the mainstream efficiency range, or that it has 47 quantized variants on the Hub. That information is trapped in model cards, naming conventions, config files, and community knowledge.

This means you can't ask questions like:
- "Most general Llama base model that supports tool-calling and fits on consumer GPU"
- "Models architecturally similar to Mamba but with instruction tuning"
- "What distinguishes these two models? What do they share?"
- "Navigate from this model toward smaller and more code-focused"

These aren't filter queries. They're **navigation** — moving through a space of related models, understanding what's near what and why.

## How It Works

Start with a single idea: every model has a **position** along several independent dimensions.

Take efficiency. A 7B model is the mainstream sweet spot — most people searching for models land here. So 7B is **zero**. Smaller models (3B, 1B) go negative. Larger models (30B, 70B) go positive. Now any query about model size is just arithmetic: "small" means "negative in EFFICIENCY."

Apply this to seven dimensions:

```
ARCHITECTURE    zero = transformer decoder          the overwhelming default
CAPABILITY      zero = general language model       before specialization
EFFICIENCY      zero = ~7B parameters               the mainstream sweet spot
COMPATIBILITY   zero = PyTorch + transformers        universal baseline
LINEAGE         zero = base/foundational model      before fine-tuning
DOMAIN          zero = general knowledge             before domain narrowing
QUALITY         zero = established, mainstream       known and stable
```

Negative means simpler, earlier, more general. Positive means more specialized, derived, novel. Zero is always the **most common thing people look for** — so most queries resolve near the origin.

This is the first layer: **structured coordinates in model space.**

The second layer is the **anchor dictionary** — a shared vocabulary of characteristics. "instruction-following", "tool-calling", "GGUF-available", "Apple-Silicon-native", "Llama-family." Models link to anchors. Two models sharing 15 anchors are similar, without anyone wiring up an explicit edge between them. Similarity is **emergent**.

Because anchors are sets, you get set operations for free:
- **Intersection** of two models' anchors = what they share
- **Symmetric difference** = what makes them different
- **Jaccard similarity** = overall semantic overlap

Every similarity score traces back to specific shared anchors and bank positions. Nothing is an opaque embedding.

The third layer is **spreading activation**: given a seed model, activation propagates through explicit links (fine-tuned-from, same-family) and shared anchors, decaying with distance. "Models like X" finds both direct relatives and structurally similar models from other families.

## What This Is Not

- **Not a vector store.** No embeddings. Similarity comes from shared structure, not cosine distance in a latent space.
- **Not a SQL database with 65 columns.** The seven banks and anchor dictionary replace flat attributes with navigable dimensions.
- **Not a HuggingFace API wrapper.** HF is one data source. The value is the extracted structure HF doesn't provide.
- **Not a ranking system.** There's no "best model" score. There's "what's near this point in model space, and what path leads toward what you need."

## Quick Start

```bash
uv sync
uv run model-atlas   # starts MCP server
```

Registered as a global MCP server — available in every Claude Code session automatically.

## Tools

**`hf_search_models`** — Compound navigational search. Parses natural language into bank constraints + anchor targets, scores via four channels (bank proximity, anchor Jaccard, spreading activation, fuzzy matching).

```
"small code model with tool-calling"
 → EFFICIENCY < 0, anchors: code-generation + tool-calling
```

**`hf_build_index`** — Fetch models from a source (HuggingFace, Ollama, or both), extract bank positions and anchors, add to the network. Additive — each call enriches the same graph.

**`hf_get_model_detail`** — Full semantic profile: all 7 bank positions, complete anchor set, lineage links, overflow metadata.

**`hf_compare_models`** — Anchor set operations between models. Shared characteristics, distinguishing features, per-bank position deltas.

**`set_model_vibe`** — The calling LLM writes a one-sentence vibe summary and optional anchors after reading a model card. The LLM *is* the NLP extraction tier.

**`hf_index_status`** — Network stats: model count, anchor dictionary size, per-bank coverage.

## Storage

| Path | Contents |
|------|----------|
| `~/.cache/model-atlas/network.db` | Semantic network (SQLite) |
| `~/.cache/model-atlas/extraction_cache/` | Cached raw API responses |

## Design Reference

Full system design document: [`.claude/CLAUDE.md`](.claude/CLAUDE.md)

Theoretical foundation (signed hierarchies, anchor dictionaries, spreading activation): [Sparse Wiki Grounding](https://github.com/rohanvinaik/sparse-wiki-grounding)
