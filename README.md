# ModelAtlas

[![CI](https://github.com/rohanvinaik/ModelAtlas/actions/workflows/ci.yml/badge.svg)](https://github.com/rohanvinaik/ModelAtlas/actions/workflows/ci.yml)
[![Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_ModelAtlas&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_ModelAtlas)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_ModelAtlas&metric=coverage)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_ModelAtlas)
[![Maintainability](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_ModelAtlas&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_ModelAtlas)
[![Reliability](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_ModelAtlas&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_ModelAtlas)

Have you ever spent a wasted afternoon digging through HuggingFace, looking for that model you saw last week that was absolutely *perfect* for your experiment, but now it's buried under 62 different versions of some fancy new model that has been fine tuned beyond all sense of decency? I have. Ever waste a week on some inferior model that an LLM recommended, only to stumble upon the *perfect* model posted by @FartKnocker6969 on Twitter? No comment.

Regardless, it's clear that the field is starting to suffer from its own success; the profusion of models that are small enough to fit on consumer hardware has made finding the *right* model almost impossible. HF's model search is...adequate. If you know what you're looking for. But if you *don't* know what you're looking for--if you're looking to *discover* a model that fits your specific requirements--you're left navigating a jungle of LLMs without a map.

ModelAtlas is the map.

A structured semantic network of ML models. Built with simple symbolic operations. Typed, structural data encoding of relative model characteristics, with a tiny footprint and searches executed at the speed of thought. All exposed as an MCP tool, so the LLM you're already talking to can *see* the model landscape itself. Get a subconscious "vibe" of model architectures, and help you find the models you didn't know you were searching for.

### See the difference

You want a small code model with tool-calling that runs on consumer hardware.

**HuggingFace** gives you the most popular code models, regardless of size:

```
Qwen/Qwen2.5-Coder-32B-Instruct          847K downloads
Qwen/Qwen3-Coder-480B-A35B-Instruct       75K downloads
Qwen/Qwen3-Coder-Next                      1.1M downloads
Phind/Phind-CodeLlama-34B-v2                2K downloads
```

32B. 480B. Not small. HuggingFace can filter by tag and sort by popularity, but it can't express "small" as a *direction* or "tool-calling" as a *capability* — so it gives you the biggest, most popular code models instead.

**ModelAtlas** navigates to exactly what you asked for:

```python
navigate_models(efficiency=-1, capability=+1,
                require_anchors=["code-generation"],
                prefer_anchors=["instruction-following", "tool-calling"])
```

```
LiquidAI/LFM2.5-1.2B-Instruct-GGUF        1B  | code-generation, function-calling, GGUF
LocoreMind/LocoOperator-4B                  3B  | code-generation, function-calling, GGUF
Manojb/Qwen3-4B-toolcalling-gguf-codex     3B  | code-generation, function-calling, GGUF
adityakum667388/lumichat_coder-v2.1         3B  | code-generation, consumer-GPU-viable
codelion/Llama-3.2-1B-Instruct-tool-calling 1B  | code-generation, function-calling
```

1B-3B models. Code generation. Tool-calling. Consumer-GPU-viable. GGUF-ready. Every result is a direct hit — not because of keyword matching, but because ModelAtlas has a coordinate system that knows what "small" and "capable" mean as *positions in model space*.

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

## Beta status

ModelAtlas is in active beta. The semantic network contains **19,498 models** with **166 anchors** across all 8 banks and **128K+ model-anchor links**. The top-downloaded models on HuggingFace are all present and enriched.

**What works today:**
- Navigation queries return meaningfully different results than keyword search. "Small code models with tool-calling" surfaces LiquidAI LFM-1.2B, Qwen3-4B tool-calling variants, and Llama-3.2-1B function-calling adapters — real answers to a query HuggingFace can't express.
- The 8-bank coordinate system captures structural relationships that tags and filters miss. Signed directions, anchor set intersections, and IDF-weighted similarity all function as described.
- ~7,300 models have LLM-generated enrichment (summaries + capability anchors) beyond deterministic extraction. The remaining ~12K have full Tier 1+2 structural data.

**Validation:**
- ~3,000 models independently validated by Gemini against raw HuggingFace metadata.
- A multi-phase correction pipeline (deterministic audit, dictionary expansion, LLM healing) is actively improving anchor accuracy, with a target of 90-95%+ in the final network.

**What this means for users:**
The network is directionally correct and dense enough for the core use case: giving an LLM a structural sense of model space it doesn't have in its weights. Popular models are well-covered. The long tail is still being refined. Expect anchor accuracy to improve steadily as the correction pipeline converges.

## Quick start

**1. Install:**

```bash
uv sync
```

**2. Download the pre-built network:**

The semantic network is distributed as a SQLite file attached to [GitHub Releases](https://github.com/rohanvinaik/ModelAtlas/releases). Download the latest `network.db` and place it in the cache directory:

```bash
# Download latest release (update URL for current version)
curl -L -o ~/.cache/model-atlas/network.db \
  https://github.com/rohanvinaik/ModelAtlas/releases/latest/download/network.db
```

Or manually: go to [Releases](https://github.com/rohanvinaik/ModelAtlas/releases), download `network.db`, and move it to `~/.cache/model-atlas/`.

**3. Run:**

```bash
uv run model-atlas
```

**4. Add to your MCP client:**

For Claude Desktop, add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "model-atlas": {
      "command": "uv",
      "args": ["--directory", "/path/to/ModelAtlas", "run", "model-atlas"]
    }
  }
}
```

For Claude Code, add to `.mcp.json` in your project:

```json
{
  "mcpServers": {
    "model-atlas": {
      "command": "uv",
      "args": ["--directory", "/path/to/ModelAtlas", "run", "model-atlas"]
    }
  }
}
```

That's it. Your LLM can now see the model landscape.

Without the pre-built network, ModelAtlas starts with an empty database. You can build your own using `hf_build_index`, but the pre-built network includes 19K+ models with multi-tier extraction already applied.

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
| `search_models` | Search models across multiple sources (HuggingFace, Ollama, or all) |
| `list_model_sources` | List available model sources and their connection status |

## How it works underneath

**Extraction** runs in three tiers:

1. **Deterministic** — parameter count, architecture type, download velocity. Pure arithmetic on structured API fields.
2. **Pattern matching** — regex on tags, model names, configs. Detects instruction-tuning, quantization formats, family membership, domain signals.
3. **Vibe extraction** — a small (sub-7B) local model produces a one-sentence summary and extra anchors via constrained generation. Runs once per model during ingestion.

**Ingestion** is additive. Each `hf_build_index` call enriches the same network. A background daemon can run continuously, streaming new models through all three extraction tiers.

**Storage** is `~/.cache/model-atlas/network.db` — one SQLite file.

Execution of the query is done with *pure* symbolic processes. Jaccard similarity. Logarithmic decay. Signed integer directional traversal. Basic set theory operations. Math--not inference. 

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

## Data distribution

The semantic network (`network.db`) is distributed separately from the code. The database is ~80MB and contains the full model graph, anchor dictionary, and metadata.

**GitHub Releases** (current): Each beta release includes `network.db` as a release asset. Download once, place in `~/.cache/model-atlas/`, and the MCP server picks it up automatically. New releases are cut as the correction pipeline improves accuracy and coverage.

**HuggingFace Dataset** (planned): The network will also be published as a HuggingFace dataset for discoverability within the ML community. This gives download stats, versioned snapshots, and a familiar interface for ML practitioners. The dataset will include `network.db` plus a metadata card documenting anchor dictionary coverage, validation metrics, and extraction provenance.

See [`docs/data-distribution.md`](docs/data-distribution.md) for release procedures and versioning policy.

## Design reference

- Theory and design: [`docs/DESIGN.md`](docs/DESIGN.md)
- Pipeline reference: [`docs/pipeline.md`](docs/pipeline.md)
- Data distribution: [`docs/data-distribution.md`](docs/data-distribution.md)
- Architectural spec: [`.claude/CLAUDE.md`](.claude/CLAUDE.md)
- Theoretical foundation: [Sparse Wiki Grounding](https://github.com/rohanvinaik/sparse-wiki-grounding)
