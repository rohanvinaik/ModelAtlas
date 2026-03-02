# ModelAtlas

[![CI](https://github.com/rohanvinaik/ModelAtlas/actions/workflows/ci.yml/badge.svg)](https://github.com/rohanvinaik/ModelAtlas/actions/workflows/ci.yml)
[![Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_ModelAtlas&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_ModelAtlas)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_ModelAtlas&metric=coverage)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_ModelAtlas)
[![Maintainability](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_ModelAtlas&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_ModelAtlas)
[![Reliability](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_ModelAtlas&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_ModelAtlas)

Have you ever spent a wasted afternoon digging though HuggingFace, looking for that model you saw last week that was absolutely *perfect* for your experiment, but now it's burried under 62 different versions of some fancy new model that has been fine tuned beyond all sense of decency? I have. Ever waste a week on some inferior model that an LLM recommended, only to stumble upon the *perfect* model posted by @FartKnocker6969 on Twitter? No comment. 

Regardless, it's clear that the field is starting to suffer from its own success; the profusion of models that are small enough to fit on consumer hardware has made finding the *right* model almost impossible. HF's model search is...adequate. If you know what you're looking for. But if you *don't* know what you're looking for--if you're looking to *discover* a model that fits your specific requirements--you're left navigating a jungle of LLMs without a map.

ModelAtlas is the map. 

A structured semantic network of ML models. Built with simple symbolic operations. Typed, structural data encoding of relative model characteristics, with a tiny footprint and searches executed at the speed of thought. All exposed as an MCP tool, so the LLM you're already talking to can *see* the model landscape itself. Get a subconscious "vibe" of model architectures, and help you find the models you didn't know you were searching for.

## The gap

HuggingFace knows that `meta-llama/Llama-3.1-8B-Instruct` has 42,000 likes and uses the `transformers` library. What it doesn't know: this model is an instruction-tuned derivative of a base model in the Llama family, supports tool-calling, sits in the mainstream efficiency range, and has 47 quantized variants on the Hub. That information exists — scattered across model cards, naming conventions, config files, and community knowledge. But it's not queryable.

There isn't an API call or a search bar that answers:

- "What's the most general Llama base that supports tool-calling and fits on consumer GPU?"
- "What are some models that arearchitecturally similar to Mamba, but with instruction tuning?"
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

## Quick start

```bash
uv sync
uv run model-atlas
```

MCP server — available in any model that follows the standard.

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
| `hf_get_model_detail` | Full semantic profile: all 8 bank positions, anchor set, lineage, metadata |
| `hf_compare_models` | Set operations on anchor sets: shared features, distinguishing features, Jaccard similarity |
| `set_model_vibe` | LLM writes a one-sentence vibe summary after reading a model card |
| `hf_index_status` | Network stats |

## How it works underneath

**Extraction** runs in three tiers:

1. **Deterministic** — parameter count, architecture type, download velocity. Pure arithmetic on structured API fields.
2. **Pattern matching** — regex on tags, model names, configs. Detects instruction-tuning, quantization formats, family membership, domain signals.
3. **Vibe extraction** — a small (sub-7B) local model produces a one-sentence summary and extra anchors via constrained generation. Runs once per model during ingestion.

**Ingestion** is additive. Each `hf_build_index` call enriches the same network. A background daemon can run continuously, streaming new models through all three extraction tiers.

**Storage** is `~/.cache/model-atlas/network.db` — one SQLite file.

Execution of the query is done with *pure* symbolic processes. Jaccard similarity. Logmarithic decay. Signed integer directional traversal. Basic set theory operations. Math--not inference. 

Don't waste tokens on problems that have been solved for 50 years--waste them on *your* terms.

## Ingestion pipeline

A multi-phase background pipeline populates the semantic network:

| Phase | What | Scale |
|-------|------|-------|
| **A: Fetch** | Stream HF API, cache raw JSON + config.json + model cards | ~40K models |
| **B: Extract** | Deterministic + pattern matching (Tier 1+2) | ~38K extracted |
| **C1: Summarize** | Smol-Hub-tldr (360M) on card text | ~7K models, ~1 hour |
| **C2: Structure** | qwen2.5:3b via Ollama — summary + anchors | ~38K models, ~3 days |
| **C3: Quality gate** | Blind review of generated outputs | ~38K models, ~3 days |
| **C4: Validate** | Offline comparison vs ground truth datasets | Seconds |

C1 and C2 run in parallel on different resources (transformers vs Ollama). Summary selection prefers Smol-Hub-tldr summaries when available (purpose-built for model cards) and falls back to qwen2.5:3b output. All workers are standalone scripts deployable to any machine via scp.

See [`docs/pipeline.md`](docs/pipeline.md) for the full operational reference.

## Design reference

- Theory and design: [`docs/DESIGN.md`](docs/DESIGN.md)
- Pipeline reference: [`docs/pipeline.md`](docs/pipeline.md)
- Architectural spec: [`.claude/CLAUDE.md`](.claude/CLAUDE.md)
- Theoretical foundation: [Sparse Wiki Grounding](https://github.com/rohanvinaik/sparse-wiki-grounding)
