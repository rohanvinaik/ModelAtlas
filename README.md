# ModelAtlas

**Navigate HuggingFace's 800K models by semantic coordinates, not keywords.**

[![CI](https://github.com/rohanvinaik/ModelAtlas/actions/workflows/ci.yml/badge.svg)](https://github.com/rohanvinaik/ModelAtlas/actions/workflows/ci.yml)
[![Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_ModelAtlas&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_ModelAtlas)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_ModelAtlas&metric=coverage)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_ModelAtlas)
[![Maintainability](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_ModelAtlas&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_ModelAtlas)
[![Reliability](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_ModelAtlas&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_ModelAtlas)

HuggingFace has 800K models and no way to ask "find me a small code model with tool-calling." So I built a coordinate system.

19,498 models positioned across 8 semantic dimensions. No embeddings, no GPU, no API calls at query time. Pure arithmetic on a SQLite file.

---

You want a small code model with tool-calling.

**HuggingFace** gives you the biggest, most popular code models:

```
Qwen/Qwen2.5-Coder-32B-Instruct          847K downloads
Qwen/Qwen3-Coder-480B-A35B-Instruct       75K downloads
Qwen/Qwen3-Coder-Next                      1.1M downloads
```

32B. 480B. Not small. HF can filter by tag and sort by popularity, but it can't express "small" as a *direction*.

**ModelAtlas** navigates to what you actually asked for:

```python
navigate_models(efficiency=-1, capability=+1,
                require_anchors=["code-generation"],
                prefer_anchors=["tool-calling"])
```

```
LiquidAI/LFM2.5-1.2B-Instruct-GGUF        1B  | code-generation, function-calling, GGUF
LocoreMind/LocoOperator-4B                  3B  | code-generation, function-calling, GGUF
Manojb/Qwen3-4B-toolcalling-gguf-codex     3B  | code-generation, function-calling, GGUF
codelion/Llama-3.2-1B-Instruct-tool-calling 1B  | code-generation, function-calling
```

Every result is a direct hit. Not keyword matching — *position in model space*.

<!-- TODO: mlx-vis 2D projection of the semantic network (19K models, colored by domain) -->

---

## How it works

Eight signed dimensions. Each has a zero state — the most common thing people look for.

```
ARCHITECTURE    zero = transformer decoder       →  +novel (Mamba, MoE)
CAPABILITY      zero = general language model     →  +rich (code, tools, reasoning)
EFFICIENCY      zero = ~7B parameters             →  +larger  / -smaller
COMPATIBILITY   zero = PyTorch + transformers     →  +specific (GGUF, MLX)
LINEAGE         zero = base/foundational model    →  +derived (fine-tune, quant)
DOMAIN          zero = general knowledge           →  +specialized (code, medical)
QUALITY         zero = established mainstream      →  +trending  / -legacy
TRAINING        zero = standard supervised (SFT)  →  +complex (RLHF, DPO) / -simpler
```

On top of coordinates, models share **anchors** — labels like "instruction-following", "GGUF-available", "Llama-family." Similarity is emergent from shared labels, weighted by rarity (IDF). Every score traces back to specific anchors. Nothing is an opaque embedding.

**Scoring:** `bank_alignment × anchor_relevance × seed_similarity`. Multiplicative — a model that nails efficiency but misses capability gets zero, not fifty percent.

**Extraction** runs in three tiers: deterministic (API fields, parameter math) → pattern matching (tags, names, configs) → vibe extraction (small local LLM, once per model at ingestion). At query time, it's multiplication and set intersection. Math — not inference.

## What this is not

- **Not a vector store.** No embeddings. Similarity comes from shared structure.
- **Not a HuggingFace wrapper.** HF is a data source. The value is the extracted structure HF doesn't expose.
- **Not a ranking system.** No "best model" score. Navigation, not leaderboard.

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/rohanvinaik/ModelAtlas.git && cd ModelAtlas && uv sync

# 2. Download pre-built network (19K+ models, all extraction tiers applied)
mkdir -p ~/.cache/model-atlas
curl -L -o ~/.cache/model-atlas/network.db \
  https://github.com/rohanvinaik/ModelAtlas/releases/latest/download/network.db

# 3. Add to Claude Code (.mcp.json) or Claude Desktop config
```

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

That's it. Your LLM can now see model space.

## Tools

| Tool | Purpose |
|------|---------|
| `navigate_models` | Primary. Bank directions + anchor targeting → ranked results |
| `hf_search_models` | Natural language fallback with fuzzy matching |
| `hf_get_model_detail` | Full semantic profile: all 8 positions, anchors, lineage |
| `hf_compare_models` | Structural diff via anchor set operations + Jaccard similarity |
| `hf_build_index` | Ingest models from HuggingFace/Ollama into the network |
| `set_model_vibe` | LLM-generated one-sentence model summary |
| `search_models` | Multi-source search (HuggingFace, Ollama, or all) |
| `list_model_sources` | Available sources and connection status |
| `hf_index_status` | Network statistics |

<!-- TODO: ## Performance

| Metric | Value |
|--------|-------|
| Query latency (p50) | TBD ms |
| Query latency (p95) | TBD ms |
| Query latency (p99) | TBD ms |
| Network size | 19,498 models, 166 anchors, 128K+ links |
| Memory footprint | ~XX MB (SQLite + Python process) |
| Neural compute at query time | Zero |
-->

## Status

19,498 models. 166 anchors. 128K+ model-anchor links. ~7,300 models with LLM-enriched summaries; the rest have full structural data from deterministic + pattern extraction. ~3,000 models independently validated against raw HF metadata. Multi-phase correction pipeline actively converging toward 90-95%+ anchor accuracy.

The network is dense enough for the core use case: giving an LLM structural awareness of model space that isn't in its weights. Popular models are well-covered. Long tail still refining. HuggingFace has real-time download counts and community activity; ModelAtlas is a periodic snapshot — it tells you *what to look at*, not *what's trending right now*.

Part of a research program on structured navigation through constrained semantic spaces — the same paradigm applied to [theorem proving](https://github.com/rohanvinaik/Wayfinder) and [code quality supervision](https://github.com/rohanvinaik/LintGate).

## Deep dive

| | |
|---|---|
| Full docs | [rohanv.me/ModelAtlas](https://rohanv.me/ModelAtlas/) |
| Pipeline reference | [`docs/pipeline.md`](docs/pipeline.md) |
| Design deep dive | [`docs/DESIGN.md`](docs/DESIGN.md) |
| Theoretical foundation | [Sparse Wiki Grounding](https://github.com/rohanvinaik/sparse-wiki-grounding) |

---

MIT — Rohan Vinaik
