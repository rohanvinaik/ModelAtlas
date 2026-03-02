# ModelAtlas

Ask your LLM to find you a model and watch what happens. It recites whatever it memorized during training — names that were popular six months ago, vague impressions of what's good at code, no sense of what's related to what. It's navigating a landscape of a million models with no map.

ModelAtlas is the map. A structured semantic network of ML models, exposed as an MCP tool, so the LLM you're already talking to can *see* the model landscape instead of guessing at it.

## The gap

HuggingFace knows that `meta-llama/Llama-3.1-8B-Instruct` has 42,000 likes and uses the `transformers` library. What it doesn't know: this model is an instruction-tuned derivative of a base model in the Llama family, supports tool-calling, sits in the mainstream efficiency range, and has 47 quantized variants on the Hub. That information exists — scattered across model cards, naming conventions, config files, and community knowledge. But it's not queryable.

So you can't ask:

- "Most general Llama base that supports tool-calling and fits on consumer GPU"
- "Models architecturally similar to Mamba but with instruction tuning"
- "Navigate from here toward smaller and more code-focused"

These aren't filter queries. They're **navigation** — and HuggingFace doesn't have a coordinate system to navigate with.

## The idea

Every model has a position along seven independent dimensions. Take efficiency: 7B is the mainstream sweet spot, so 7B is **zero**. Smaller goes negative. Larger goes positive. "Small" just means "negative in EFFICIENCY."

```
ARCHITECTURE    zero = transformer decoder       →  +novel (Mamba, MoE)
CAPABILITY      zero = general language model     →  +rich (code, tools, reasoning)
EFFICIENCY      zero = ~7B parameters             →  +larger  / -smaller
COMPATIBILITY   zero = PyTorch + transformers     →  +specific (GGUF, MLX)
LINEAGE         zero = base/foundational model    →  +derived (fine-tune, quant)
DOMAIN          zero = general knowledge           →  +specialized (code, medical)
QUALITY         zero = established mainstream      →  +trending  / -legacy
```

Zero is always **the most common thing people look for**. Most queries resolve near the origin.

On top of coordinates, models share **anchors** — a vocabulary of characteristics like "instruction-following", "GGUF-available", "Llama-family." Models sharing anchors are similar without explicit edges. Similarity is emergent, and every score traces back to specific shared labels. Nothing is an opaque embedding.

The LLM decomposes a user's question into coordinates and anchors. ModelAtlas does arithmetic on integers and set intersections on small lists. The intelligence is in the interaction — no single piece is smart, but the system is.

## What this is not

- **Not a vector store.** No embeddings. Similarity comes from shared structure.
- **Not a database with 65 columns.** Seven signed dimensions and a label vocabulary replace flat attributes.
- **Not a HuggingFace wrapper.** HF is a data source. The value is the extracted structure HF doesn't expose.
- **Not a ranking system.** No "best model" score. Just "what's near here, and what path leads where you need."

The entire thing is a SQLite file, a few thousand anchor labels, and signed integers. No GPU at query time. No vector store in the background. No running services. The heaviest operation is optional vibe extraction during ingestion (a 0.5B model, runs once per model, ever). At query time it's multiplication and set intersection.

## Quick start

```bash
uv sync
uv run model-atlas
```

MCP server — available in any Claude Code session.

## Usage

The primary tool is `navigate_models`. The calling LLM fills in structured parameters; ModelAtlas does deterministic scoring.

```python
navigate_models(
    efficiency=-1,           # small
    capability=1,            # capable
    require_anchors=["code-generation"],
    prefer_anchors=["instruction-following", "tool-calling"],
    avoid_anchors=["embedding"]
)
# → Small code models with tool-calling, ranked by IDF-weighted anchor overlap
```

**Bank directions** — `-1`, `0`, or `+1` per dimension. Omit any you don't care about.

**Anchor targeting** — `require` (hard filter), `prefer` (IDF-weighted boost — rare anchors count more), `avoid` (each match halves the score).

**Seed similarity** — `similar_to="meta-llama/Llama-3.1-8B-Instruct"` finds models with overlapping anchor sets, weighted by anchor rarity.

**Scoring** — `bank_alignment * anchor_relevance * seed_similarity`. Multiplicative: a model that nails efficiency but misses capability gets zero, not fifty percent.

### Other tools

| Tool | What it does |
|------|-------------|
| `hf_search_models` | Natural language fallback — keyword parsing into bank constraints + fuzzy matching |
| `hf_build_index` | Fetch models from HuggingFace/Ollama, extract positions and anchors, add to network |
| `hf_get_model_detail` | Full semantic profile: all 7 bank positions, anchor set, lineage, metadata |
| `hf_compare_models` | Set operations on anchor sets: shared features, distinguishing features, Jaccard similarity |
| `set_model_vibe` | LLM writes a one-sentence vibe summary after reading a model card |
| `hf_index_status` | Network stats |

## How it works underneath

**Extraction** runs in three tiers:

1. **Deterministic** — parameter count, architecture type, download velocity. Pure arithmetic on structured API fields.
2. **Pattern matching** — regex on tags, model names, configs. Detects instruction-tuning, quantization formats, family membership, domain signals.
3. **Vibe extraction** — a 0.5B local model produces a one-sentence summary and extra anchors via constrained generation. Runs once per model during ingestion.

**Ingestion** is additive. Each `hf_build_index` call enriches the same network. A background daemon can run continuously, streaming new models through all three extraction tiers.

**Storage** is `~/.cache/model-atlas/network.db` — one SQLite file.

## Design reference

- Theory and design: [`docs/DESIGN.md`](docs/DESIGN.md)
- Architectural spec: [`.claude/CLAUDE.md`](.claude/CLAUDE.md)
- Theoretical foundation: [Sparse Wiki Grounding](https://github.com/rohanvinaik/sparse-wiki-grounding)
