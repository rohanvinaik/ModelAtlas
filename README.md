# ModelAtlas

**Navigate HuggingFace's 800K models by semantic coordinates, not keywords.**

[![CI](https://github.com/rohanvinaik/ModelAtlas/actions/workflows/ci.yml/badge.svg)](https://github.com/rohanvinaik/ModelAtlas/actions/workflows/ci.yml)
[![Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_ModelAtlas&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_ModelAtlas)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_ModelAtlas&metric=coverage)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_ModelAtlas)
[![Maintainability](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_ModelAtlas&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_ModelAtlas)
[![Reliability](https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_ModelAtlas&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=rohanvinaik_ModelAtlas)

HuggingFace has 800K models and no way to ask "find me a small code model with tool-calling." So I built a coordinate system.

29,657 models positioned across 8 semantic dimensions. No embeddings, no GPU, no API calls at query time. Pure arithmetic on a SQLite file.

---

You want a small code model with tool-calling.

**HuggingFace** gives you the biggest, most popular code models:

```
Qwen/Qwen2.5-Coder-32B-Instruct          32B   1,996 likes
Qwen/Qwen3-Coder-480B-A35B-Instruct     480B   1,315 likes
Qwen/Qwen3-Coder-30B-A3B-Instruct        30B     981 likes
```

32B. 480B. Not small. HF can filter by tag and sort by popularity, but it can't express "small" as a *direction*.

**ModelAtlas** navigates to what you actually asked for:

```python
navigate_models(efficiency=-1, capability=+1, domain=+1, quality=+1,
                require_anchors=["code-generation"],
                prefer_anchors=["tool-calling", "GGUF-available", "high-downloads", "trending"])
```

```
bullpoint/Qwen3-Coder-Next-AWQ-4bit        3B  | code, tool-calling, trending     score: 0.79
LocoreMind/LocoOperator-4B                  4B  | code, tool-calling, GGUF         score: 0.63
Qwen/Qwen2.5-Coder-0.5B-Instruct         0.5B  | code, high-downloads             score: 0.37
```

Every result is a direct hit — small, code-focused, tool-calling, popular. Not keyword matching — *position in model space*.

### Head to head with HuggingFace

Same queries, run through both systems (March 2026). HuggingFace uses its API with the best available `pipeline_tag` filters and sort-by-likes. ModelAtlas uses `navigate_models` with `quality=+1` for light popularity weighting. All results are real and reproducible.

**Level 1 — Common queries.** Both systems should get these right. The critical test: *does ModelAtlas reproduce the known-good answers?*

| Query | HuggingFace | ModelAtlas | Verdict |
|-------|-------------|-----------|---------|
| **Sentiment analysis** | cardiffnlp/twitter-roberta-base-sentiment (782 likes), nlptown/bert-multilingual-sentiment (466) | cardiffnlp/twitter-roberta-base-sentiment, nlptown/bert-multilingual-sentiment, ProsusAI/finbert | **Same top results** + finbert as a bonus |
| **Named entity recognition** | dslim/bert-base-NER (702), blaze999/Medical-NER (225) | dslim/bert-base-NER, microsoft/deberta-v3-base, FacebookAI/xlm-roberta-conll03 | **Same #1** + stronger alternatives |
| **Image captioning** | Salesforce/blip-image-captioning-large (1,458), vit-gpt2 (927) | OpenGVLab/InternVL2-2B, Qwen2-VL-2B-Instruct, Qwen2.5-VL-3B-Instruct | **Both excellent** — HF returns classic captioners, MA returns modern VLMs |

ModelAtlas matches HuggingFace beat-for-beat on simple queries. This is the baseline. If it couldn't do this, nothing else would matter.

**Level 2 — Directional queries.** These have a concept ("small," "fast," "medical") that HF can only approximate with keyword matching.

| Query | HuggingFace | ModelAtlas | Verdict |
|-------|-------------|-----------|---------|
| **Small code model** | codeparrot-small (33 likes), CodeGPT-small-py (28). Ancient, tiny models nobody uses. | Qwen3-Coder-Next-AWQ-4bit (3B, trending), Qwen2.5-Coder-0.5B (high-downloads) | **MA wins** — current models, right size |
| **Fast embedding model** | *No results.* "Fast" isn't a tag. | Qwen3-Embedding-0.6B (high-downloads), Octen-Embedding-0.6B (edge-deployable), jina-v5-text-small | **MA wins** — sub-1B, edge-deployable |
| **Medical text classifier** | medical_o1_verifier_3B (a reasoning *verifier*, not a classifier), Bio_ClinicalBERT (15 likes) | StanfordAIMI/stanford-deidentifier-base (high-downloads, encoder-only), obi/deid_bert_i2b2, obi/deid_roberta_i2b2 | **MA wins** — actual classifiers from clinical NLP groups |

HuggingFace starts breaking. "Small" maps to models with "small" in the name (from 2021). "Fast" returns nothing. "Medical classifier" returns a reasoning verifier. ModelAtlas returns what you meant, not what you typed.

**Level 3 — Multi-constraint queries.** These combine direction + domain + negation. HuggingFace cannot express them at all.

| Query | HuggingFace | ModelAtlas |
|-------|-------------|-----------|
| **Small multilingual chat model** | *No results.* | PaddleOCR-VL-1.5 (sub-1B, multilingual, edge), Nanbeige4.1-3B-GGUF, LFM2-2.6B-GGUF |
| **Tiny speech model for on-device** | *No results.* | Qwen3-TTS-1.7B-VoiceDesign, granite-4.0-1b-speech, **MioTTS-0.1B** (100M params!) |
| **Legal document model with multilingual** | 1 result (1 like) | Arabic legal OCR (3B, trending), Taiwan legal+medical (3B, Llama), next-1b (legal+finance+medical, 1B) |

### Queries that make HuggingFace impossible

**"Multilingual chat model for edge — NOT code, NOT math, NOT embedding."** Try this on HuggingFace. There is no combination of tags and filters that expresses "multilingual AND chat AND small AND NOT three specific domains." You'd search "multilingual chat" and scroll through thousands of results, manually skipping every code model.

```python
navigate_models(efficiency=-1, domain=+1, quality=+1,
                require_anchors=["multilingual"],
                prefer_anchors=["edge-deployable", "consumer-GPU-viable", "GGUF-available"],
                avoid_anchors=["code-domain", "math-domain", "embedding"])
```

```
PaddlePaddle/PaddleOCR-VL-1.5       sub-1B  | multilingual, edge-deployable, trending    0.85
Edge-Quant/Nanbeige4.1-3B-GGUF         3B   | multilingual, GGUF, llama.cpp              0.60
LiquidAI/LFM2-2.6B-Exp-GGUF          2.6B   | multilingual, GGUF, llama.cpp              0.60
```

**"Tiny speech model for on-device TTS."** HuggingFace returns nothing useful for "small speech edge." ModelAtlas finds a **100-million-parameter** TTS model:

```python
navigate_models(efficiency=-1, capability=-1, domain=+1, quality=+1,
                require_anchors=["speech-domain"],
                prefer_anchors=["consumer-GPU-viable", "edge-deployable", "trending", "multilingual"])
```

```
Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign  1.7B  | speech, multilingual, trending             0.31
ibm-granite/granite-4.0-1b-speech        1B  | speech, multilingual, encoder-decoder       0.31
Aratako/MioTTS-0.1B                    0.1B  | speech, edge-deployable, sub-1B             0.25
FunAudioLLM/Fun-CosyVoice3-0.5B       0.5B  | speech, ONNX, edge-deployable               0.25
```

**"Biology classifier, small, encoder architecture."** This surfaces models that maybe 20 people in the world are looking for at any given time:

```python
navigate_models(efficiency=-1, domain=+1, quality=+1,
                require_anchors=["biology-domain"],
                prefer_anchors=["classification", "consumer-GPU-viable", "encoder-only"])
```

```
microsoft/BiomedNLP-BiomedBERT           base  | biology+chemistry+medical, encoder-only   0.15
Ihor/gliner-biomed-large-v1.0           large  | biomedical NER, encoder-only, classification
MilosKosRad/BioNER                       base  | biological NER, encoder-only
PoetschLab/GROVER                        base  | genomics, encoder-only, classification
```

PoetschLab/GROVER is a *genomics* model. It has 7 anchors. It exists on HuggingFace with minimal visibility. ModelAtlas found it because `biology-domain` + `classification` + `encoder-only` is a precise intersection, not a keyword.

**The pattern:** The harder the query, the wider the gap. Simple queries — both work. Directional queries — MA wins. Multi-constraint queries — HF returns nothing; MA finds exactly what you need.

But the Level 1 result is arguably the most important. Anyone can build a weird niche search system. Building one that *also matches the incumbent beat-for-beat on common queries* — that's what makes it a replacement, not a toy.

---

## What the LLM actually gets

When an LLM calls `navigate_models`, it doesn't just get a list of model IDs. It gets a **structural understanding of each model's position in semantic space** — what it's for, how it relates to other models, why it scored the way it did.

A single tool call returns:

```json
{
  "model_id": "ibm-granite/granite-3b-code-instruct-128k",
  "score": 0.86,
  "score_breakdown": {"bank_alignment": 1.0, "anchor_relevance": 0.86},
  "positions": {"CAPABILITY": "+3", "EFFICIENCY": "-1", "DOMAIN": "+1"},
  "anchors": ["code-generation", "tool-calling", "long-context", "math", "consumer-GPU-viable"]
}
```

From this, the LLM knows: *This is a small, code-focused model with tool-calling and math capability, designed for consumer hardware, with 128K context.* That's not metadata — that's **model identity compressed into a structured representation the LLM can reason over.** The anchors are a vibe check. The positions are a capability profile. The score breakdown explains *why this model and not another.*

Without ModelAtlas, the LLM has whatever is frozen in its training data — model names, vague associations, outdated information. With ModelAtlas, it has a live, structured, queryable understanding of 29,657 models that costs one tool call and ~500 tokens of response. It's a subconscious understanding of the entire HuggingFace ecosystem that any model can access for the cost of a single function call.

**Cost comparison:**

| Approach | Latency | Tokens | Quality |
|----------|---------|--------|---------|
| LLM guessing from training data | 0ms | 0 | Stale, incomplete, no niche coverage |
| HuggingFace API call + parse | 2-5s | ~2,000 (API response parsing) | Tag filter + popularity sort |
| ModelAtlas `navigate_models` | <100ms | ~500 (structured JSON) | Scored, ranked, auditable, niche-aware |

The LLM doesn't need to understand model cards, parse HuggingFace HTML, or guess at model capabilities. It gets the *vibe* of every model in the network — pre-extracted, pre-scored, pre-structured — for less than the cost of a single follow-up question.

<!-- TODO: mlx-vis 2D projection of the semantic network (29K models, colored by domain) -->

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

**Scoring:** `bank_alignment × anchor_relevance × seed_similarity`. All three are multiplicative — a model that nails efficiency but misses capability gets zero, not fifty percent.

Wrong-direction models decay hyperbolically: asking for "small" gives a 7B model 0.5, a 13B model 0.5, and a 70B model 0.25. Because bank scores multiply across dimensions, a large model that's also wrong on compatibility scores `0.25 × 0.5 = 0.125` before anchors even factor in. Avoided anchors stack exponentially (each halves the score), while required anchors are hard filters — miss one and the model is invisible. The result is a scoring surface that strongly favors precise matches and rapidly eliminates mismatches, without binary cutoffs.

**Extraction** runs in three tiers: deterministic (API fields, parameter math) → pattern matching (tags, names, configs) → vibe extraction (small local LLM, once per model at ingestion). At query time, it's multiplication and set intersection. Math — not inference.

## What this is not

- **Not a vector store.** No embeddings. Similarity comes from shared structure.
- **Not a HuggingFace wrapper.** HF is a data source. The value is the extracted structure HF doesn't expose.
- **Not a ranking system.** No "best model" score. Navigation, not leaderboard.

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/rohanvinaik/ModelAtlas.git && cd ModelAtlas && uv sync

# 2. Download pre-built network (29K+ models, all extraction tiers applied)
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

29,657 models. 166 anchors. 195K model-anchor links. 26K models with prose summaries; all 29K have full structural data from deterministic + pattern extraction. 6,154 models independently validated against raw HF metadata via Gemini Pro/Flash. 700 models corrected through the D1→D3 audit/heal pipeline. 500 DPO training examples exported for next-generation extraction.

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
