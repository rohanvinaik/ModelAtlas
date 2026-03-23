# ModelAtlas

**Google for open-source AI models.** Search by what you *mean*, not by keywords.

[![CI](https://github.com/rohanvinaik/ModelAtlas/actions/workflows/ci.yml/badge.svg)](https://github.com/rohanvinaik/ModelAtlas/actions/workflows/ci.yml)
[![Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_ModelAtlas&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_ModelAtlas)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_ModelAtlas&metric=coverage)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_ModelAtlas)
[![Mutation Kill Rate](https://raw.githubusercontent.com/rohanvinaik/ModelAtlas/badges/.github/badges/mutation-kill-rate.svg)](https://github.com/rohanvinaik/ModelAtlas/actions/workflows/spec-badges.yml)
[![MC/DC](https://raw.githubusercontent.com/rohanvinaik/ModelAtlas/badges/.github/badges/mcdc.svg)](https://github.com/rohanvinaik/ModelAtlas/actions/workflows/spec-badges.yml)
[![Mean σ](https://raw.githubusercontent.com/rohanvinaik/ModelAtlas/badges/.github/badges/sigma.svg)](https://github.com/rohanvinaik/ModelAtlas/actions/workflows/spec-badges.yml)
[![Tests](https://raw.githubusercontent.com/rohanvinaik/ModelAtlas/badges/.github/badges/test-count.svg)](https://github.com/rohanvinaik/ModelAtlas/actions/workflows/spec-badges.yml)

`29,657 models · 166 semantic anchors · <100ms queries · No embeddings · No GPU`

You want a small code model with tool-calling.

**HuggingFace** gives you the biggest, most popular code models:

```
Qwen2.5-Coder-32B-Instruct          32B   1,996 likes
Qwen3-Coder-480B-A35B-Instruct     480B   1,315 likes
```

480B parameters. Not small. HF sorts by popularity. It can't express "small" as a *direction*.

**ModelAtlas** gives you what you actually asked for:

```python
navigate_models(efficiency=-1, capability=+1, quality=+1,
                require_anchors=["code-generation"],
                prefer_anchors=["tool-calling", "high-downloads"])
```

```
Qwen3-Coder-Next-AWQ-4bit        3B  | code, tool-calling, trending     0.79
LocoOperator-4B                   4B  | code, tool-calling, GGUF         0.63
Qwen2.5-Coder-0.5B-Instruct    0.5B  | code, high-downloads             0.37
```

Every result is small, code-focused, tool-calling, and popular. One tool call. ~500 tokens. <100ms.

---

## Three levels of comparison

All queries run against both systems, March 2026. HuggingFace uses its API with `pipeline_tag` filters + sort-by-likes. ModelAtlas uses `navigate_models` with `quality=+1`. All results are real.

### Level 1: ModelAtlas matches HuggingFace

Common queries where HF works well. The baseline test — *can ModelAtlas reproduce the known-good answers?*

| Query | HuggingFace | ModelAtlas |
|-------|-------------|-----------|
| Sentiment analysis | cardiffnlp/twitter-roberta-sentiment ✓ | **Same model** + ProsusAI/finbert (financial sentiment) |
| Named entity recognition | dslim/bert-base-NER ✓ | **Same model** + microsoft/deberta-v3-base |
| Image captioning | Salesforce/blip-captioning-large ✓ | OpenGVLab/InternVL2-2B, Qwen2-VL-2B ✓ |

**Both systems return the right models.** This is the most important result. Anyone can build a niche search tool. Building one that also matches the incumbent beat-for-beat on common queries is what makes it a replacement, not a toy.

### Level 2: ModelAtlas exceeds HuggingFace

Queries with direction ("small"), intent ("fast"), or domain specificity ("medical classifier") — concepts that don't map to a single HF tag.

| Query | HuggingFace | ModelAtlas |
|-------|-------------|-----------|
| Small code model | codeparrot-small (33 likes, from 2021) | Qwen2.5-Coder-0.5B-Instruct (official, high-downloads) |
| Fast embedding model | *No results* — "fast" isn't a tag | Qwen3-Embedding-0.6B, jina-v5-text-small (sub-1B, edge-deployable) |
| Medical classifier | medical_o1_verifier (a *verifier*, not a classifier) | StanfordAIMI/stanford-deidentifier-base, obi/deid_bert_i2b2 |

HuggingFace starts returning noise. "Small" matches models with "small" in the name. "Fast" returns nothing. "Medical classifier" returns a reasoning verifier. ModelAtlas returns what you *meant*, not what you *typed*.

### Level 3: ModelAtlas finds the unfindable

Multi-constraint queries with direction + domain + negation. HuggingFace cannot express these at all.

| Query | HuggingFace | ModelAtlas |
|-------|-------------|-----------|
| Multilingual chat, NOT code/math/embedding | *Impossible to express* | PaddleOCR-VL-1.5 (sub-1B), Nanbeige4.1-3B-GGUF |
| Tiny on-device TTS | *No results* | **MioTTS-0.1B** (100M params), CosyVoice3-0.5B |
| Biology classifier, encoder-only | *No results* | BiomedBERT, gliner-biomed, **PoetschLab/GROVER** (genomics) |
| Small finance classifier | *No results* — "finance" isn't a pipeline tag | **FutureMa/Eva-4B** (finance+classification, trending), DMindAI/DMind-3-mini |
| Distilled reasoning, sub-3B, NOT a fine-tune | *No results* | Qwen3.5-0.8B-Opus-Reasoning-Distilled (score: 1.0) |

A 100-million-parameter TTS model. A genomics classifier with 7 anchors. A 0.8B model distilled from Claude Opus. These models exist on HuggingFace but they are **invisible** to keyword search. ModelAtlas finds them because `biology-domain + classification + encoder-only` is a precise intersection in a coordinate system, not a string match.

**The pattern:** Simple queries → both work. Directional queries → MA wins. Multi-constraint queries → HF returns nothing; MA finds exactly what you need. The harder the question, the wider the gap.

---

## What the LLM gets

This is an MCP tool. An LLM calls it during conversation. One tool call returns:

```json
{
  "model_id": "ibm-granite/granite-3b-code-instruct-128k",
  "score": 0.86,
  "score_breakdown": {"bank_alignment": 1.0, "anchor_relevance": 0.86},
  "positions": {"CAPABILITY": "+3", "EFFICIENCY": "-1", "DOMAIN": "+1"},
  "anchors": ["code-generation", "tool-calling", "long-context", "math", "consumer-GPU-viable"]
}
```

From this, the LLM *immediately knows*: small, code-focused, tool-calling, math-capable, consumer hardware, 128K context. The anchors are a vibe. The positions are a profile. The score explains *why this model and not another.*

Without ModelAtlas, the LLM guesses from stale training data. With it, the LLM has live, structured awareness of 29,657 models for ~500 tokens — less than the cost of a follow-up question.

| Approach | Latency | Tokens | Quality |
|----------|---------|--------|---------|
| LLM guessing from training data | 0ms | 0 | Stale, incomplete, no niche coverage |
| HuggingFace API + parse | 2-5s | ~2,000 | Tag filter + popularity sort |
| **ModelAtlas** | **<100ms** | **~500** | **Scored, ranked, auditable, vibe-aware** |

---

## How it works

Eight signed dimensions. Each has a zero state — the thing most queries assume by default.

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

On top of coordinates, models share **anchors** — 166 semantic labels like `tool-calling`, `GGUF-available`, `Llama-family`. Similarity is emergent from shared labels, weighted by rarity (IDF). Every score traces back to specific anchors. Nothing is an opaque embedding.

**Scoring:** `bank_alignment × anchor_relevance × seed_similarity`. Multiplicative — a model that nails efficiency but misses capability gets zero, not fifty percent. Wrong-direction models decay hyperbolically. Avoided anchors stack exponentially (each halves the score). Required anchors are hard filters. The result is a scoring surface that strongly favors precise matches and rapidly eliminates mismatches, without binary cutoffs. [Full scoring math →](docs/DESIGN.md)

**Extraction** runs in three tiers: deterministic (API fields, parameter math) → pattern matching (tags, names, configs) → LLM classification (small local model, once per model at ingestion). At query time, it's multiplication and set intersection. Math — not inference.

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/rohanvinaik/ModelAtlas.git && cd ModelAtlas && uv sync

# 2. Download pre-built network (29K+ models, all extraction tiers applied)
mkdir -p ~/.cache/model-atlas
curl -L -o ~/.cache/model-atlas/network.db \
  https://github.com/rohanvinaik/ModelAtlas/releases/latest/download/network.db

# 3. Add to your MCP client config (Claude Code, Cursor, VS Code, etc.)
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

Works with any MCP-compatible client. Your LLM can now see model space.

## Tools

| Tool | What it does |
|------|-------------|
| `navigate_models` | **Primary.** Bank directions + anchor constraints → scored, ranked results |
| `hf_get_model_detail` | Full profile of one model: all 8 positions, anchors, lineage, metadata |
| `hf_compare_models` | Structural diff between models: shared/unique anchors, position deltas, Jaccard similarity |
| `hf_search_models` | Natural language fallback with fuzzy matching when structured query isn't needed |
| `hf_build_index` | Ingest new models from HuggingFace or Ollama into the network |
| `search_models` | Multi-source search (HuggingFace, Ollama, or all) |
| `hf_index_status` | Network statistics: model count, anchor distribution, coverage |

## What this is not

- **Not a vector store.** No embeddings. Similarity comes from shared structure.
- **Not a HuggingFace wrapper.** HF is a data source. The value is the extracted structure HF doesn't expose.
- **Not a ranking system.** No "best model" score. Navigation, not leaderboard.

## Status

29,657 models. 166 anchors. 195K model-anchor links. 26K prose summaries. 6,154 independently validated via Gemini. 700 corrected through audit/heal pipeline. Models with <10 likes are not yet indexed — the 29K represent the active, community-validated portion of HuggingFace. Periodic snapshot — tells you *what to look at*, not *what's trending right now*.

Part of a research program on structured navigation through constrained semantic spaces — the same paradigm applied to [theorem proving](https://github.com/rohanvinaik/Wayfinder) and [code quality supervision](https://github.com/rohanvinaik/LintGate).

| | |
|---|---|
| Full docs | [rohanv.me/ModelAtlas](https://rohanv.me/ModelAtlas/) |
| Pipeline reference | [`docs/pipeline.md`](docs/pipeline.md) |
| Design deep dive | [`docs/DESIGN.md`](docs/DESIGN.md) |
| Niche query showcase | [`docs/comparison.md`](docs/comparison.md) |
| Theoretical foundation | [Sparse Wiki Grounding](https://github.com/rohanvinaik/sparse-wiki-grounding) |

---

MIT — Rohan Vinaik
